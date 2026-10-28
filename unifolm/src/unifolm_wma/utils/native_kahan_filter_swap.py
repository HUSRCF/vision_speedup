"""One owner for a cached native filter and a directly witnessed Kahan GEMM."""
import ctypes as C
import hashlib
import struct
import torch
import triton
from .native_graph_world_filter_cache_probe import GEMM,NativeWorldFilterCache
from .native_sgemm_direct_params import parameters
from .native_winograd_gemm_fp16_kahan_kernel import transformed_gemm_fp16_kahan


class CachedKahanFilterSwap:
    def __init__(self,graph,weight,plan,geometry,choice,extension,api):
        assert not torch.cuda.is_current_stream_capturing()
        transform,node,old_filter,numel=plan;co,m,ci=geometry
        assert node['name']==GEMM and [a['size'] for a in node['arguments']]==[176]
        assert tuple(weight.shape)==(co,ci,3,3) and numel==36*co*ci
        payload=bytes.fromhex(node['arguments'][0]['bytes_hex']);a,b,c=struct.unpack_from('<3Q',payload)
        assert b==old_filter and b-a==36*m*ci*4 and c-b==numel*4
        assert struct.unpack_from('<3Q',payload,56)==(m*ci,ci*co,m*co)
        assert struct.unpack_from('<I',payload,152)[0]==0x3f800000
        assert struct.unpack_from('<Q',payload,160)[0]==0 and struct.unpack_from('<Q',payload,168)[0]==c
        self.graph,self.api=graph,api;self.raw,self.executable=extension.get_handles(graph)
        self.node=int(node['node'],16);self.transform_node=int(transform['node'],16)
        self.cache=NativeWorldFilterCache(graph,weight,extension,api,validated_plan=plan)
        assert not self.cache.enabled
        config=choice['config'];bm,bn,bk,warps,stages=config
        # Compile/load only. No dummy kernel launch or full-size template
        # allocations, and never use guessed native-workspace tensor wrappers.
        self.dummy=torch.empty((1,),dtype=torch.float32,device=weight.device)
        self.kernel=transformed_gemm_fp16_kahan.warmup(self.dummy,self.dummy,self.dummy,
            co,m,ci,bm,bn,bk,num_warps=warps,num_stages=stages,enable_fp_fusion=False,
            grid=(triton.cdiv(co,bm)*triton.cdiv(m,bn),36))
        self.kernel._init_handles()
        observed=hashlib.sha256(self.kernel.kernel).hexdigest()
        assert observed==choice['witness_cubin_sha256'],'Compiled kernel differs from direct-capture witness'
        self.replacement=parameters(api,self.kernel,(self.cache.cache.data_ptr(),a,c),geometry,config)
        self.closed=False;self.mode='uncached';self.enabled=False
        self.proof=dict(original_native_input_filter_output_transforms=True,
            original_filter_cloned_on_master_version=True,numerical_SGEMM_only=True,
            compensated_FP32_block_sum=True,single_executable_pointer_writer=True,
            direct_params_driver_metadata=True,direct_witness_cubin_sha256=observed,
            cached_native_filter_bytes=numel*4,kahan_template_bytes=self.dummy.numel()*4,
            primitive_strict_conv_3e6_passed=choice['primitive_strict_conv_3e6_passed'],
            diagnostic_gross_error_gate_only=True,full_video_PSNR_required=True)

    def refresh_if_needed(self):
        assert not self.closed and not self.cache.enabled
        self.cache.refresh_if_needed()

    def set_mode(self,mode):
        assert not self.closed and mode in {'uncached','cached_native','kahan'}
        assert not torch.cuda.is_current_stream_capturing()
        if mode!='uncached':self.refresh_if_needed()
        params={'uncached':self.cache.original_consumer,
            'cached_native':self.cache.cached_consumer,'kahan':self.replacement}[mode]
        try:
            self.api.check(self.api.lib.cuGraphExecKernelNodeSetParams(self.executable,self.node,
                C.byref(params.params)),'set compensated cached GEMM owner parameters')
            self.api.check(self.api.lib.cuGraphNodeSetEnabled(self.executable,self.transform_node,
                int(mode=='uncached')),'toggle native filter under compensated owner')
            enabled=C.c_uint()
            self.api.check(self.api.lib.cuGraphNodeGetEnabled(self.executable,self.transform_node,C.byref(enabled)),
                'verify native filter toggle under compensated owner')
            assert bool(enabled.value)==(mode=='uncached')
        except Exception:
            self.api.lib.cuGraphExecKernelNodeSetParams(self.executable,self.node,
                C.byref(self.cache.original_consumer.params))
            self.api.lib.cuGraphNodeSetEnabled(self.executable,self.transform_node,1)
            self.mode='uncached';self.enabled=False
            raise
        self.mode=mode;self.enabled=mode!='uncached'

    def set_enabled(self,on):self.set_mode('kahan' if on else 'uncached')
    def verify_cache(self):self.cache.verify_cache()

    def close(self):
        if self.closed:return
        torch.cuda.synchronize();self.set_mode('uncached');self.cache.close();self.closed=True
