"""One owner for native-filter caching and a four-pointer packed GEMM node."""
import ctypes as C
import struct
import torch
import triton
from .native_graph_weight_cache_probe import OwnedParams
from .native_graph_world_filter_cache_probe import GEMM,NativeWorldFilterCache
from .native_winograd_gemm_fp16_packed_kernel import pack_native_filter


def parameters4(api,kernel,pointers,geometry,config):
    assert torch.__version__=='2.3.1+cu121' and triton.__version__=='2.3.1'
    assert len(pointers)==4
    co,m,ci=geometry;bm,bn,bk,warps,stages=config
    kernel._init_handles()
    assert kernel.num_warps==warps and kernel.num_ctas==1 and tuple(kernel.cluster_dims)==(1,1,1)
    assert not kernel.metadata.get('tensormaps_info')
    for index in range(5):
        offset,size=C.c_size_t(),C.c_size_t()
        status=api.lib.cuFuncGetParamInfo(C.c_void_p(kernel.function),index,C.byref(offset),C.byref(size))
        if index==4:
            assert status==1,'Unexpected fifth packed GEMM argument'
            continue
        api.check(status,'packed GEMM bounded parameter metadata')
        assert (offset.value,size.value)==(index*8,8)
    return OwnedParams(dict(function=hex(kernel.function),
        grid=[triton.cdiv(co,bm)*triton.cdiv(m,bn),36,1],block=[32*warps,1,1],
        shared_bytes=kernel.shared,arguments=[dict(index=i,offset=i*8,size=8,
            bytes_hex=struct.pack('<Q',ptr).hex()) for i,ptr in enumerate(pointers)]))


def witness4(graph,kernel,pointers,geometry,config,extension,api):
    raw,_=extension.get_handles(graph)
    nodes=api.read(raw,{})['nodes'];assert len(nodes)==1 and nodes[0]['type']==0
    observed=OwnedParams(nodes[0]);direct=parameters4(api,kernel,pointers,geometry,config)
    for field in ['func','grid_x','grid_y','grid_z','block_x','block_y','block_z','shared']:
        assert getattr(observed.params,field)==getattr(direct.params,field)
    assert len(observed.buffers)==len(direct.buffers)==4
    assert [bytes(b) for b in observed.buffers]==[bytes(b) for b in direct.buffers]
    return dict(captured_function_grid_block_shared_and_four_argument_bytes_identical=True,
                bounded_driver_four_pointer_metadata=True)


class PrepackedSgemmSwap:
    def __init__(self,graph,weight,plan,kernel,geometry,config,extension,api):
        self.graph,self.api,self.kernel=graph,api,kernel
        self.raw,self.executable=extension.get_handles(graph)
        transform,node,old_filter,numel=plan
        assert node['name']==GEMM and [a['size'] for a in node['arguments']]==[176]
        payload=bytes.fromhex(node['arguments'][0]['bytes_hex'])
        a,b,c=struct.unpack_from('<3Q',payload);co,m,ci=geometry
        assert tuple(weight.shape)==(co,ci,3,3) and numel==36*co*ci and b==old_filter
        assert b-a==36*m*ci*4 and c-b==numel*4
        assert struct.unpack_from('<3Q',payload,56)==(m*ci,ci*co,m*co)
        assert struct.unpack_from('<I',payload,152)[0]==0x3f800000
        assert struct.unpack_from('<Q',payload,160)[0]==0 and struct.unpack_from('<Q',payload,168)[0]==c
        self.node=int(node['node'],16);self.transform_node=int(transform['node'],16)
        self.original=OwnedParams(node)
        self.cache=NativeWorldFilterCache(graph,weight,extension,api,validated_plan=plan)
        # Never enable this cache object's consumer replacement. This object
        # alone coordinates the executable filter toggle and packed GEMM.
        assert not self.cache.enabled
        self.guard=4096
        self.wh_backing=torch.full((numel+2*self.guard,),123.5,device=weight.device,dtype=torch.float16)
        self.wl_backing=torch.full_like(self.wh_backing,123.5)
        self.wh=self.wh_backing[self.guard:-self.guard]
        self.wl=self.wl_backing[self.guard:-self.guard]
        self.packed_version=None;self.refreshes=0;self.enabled=False;self.closed=False
        self.refresh_if_needed()
        self.replacement=parameters4(api,kernel,(self.wh.data_ptr(),self.wl.data_ptr(),a,c),geometry,config)
        self.proof=dict(original_native_input_filter_output_transforms=True,
            original_filter_cloned_on_master_version=True,numerical_SGEMM_only=True,
            prepacked_FP16_high_and_scaled_residual=True,single_executable_pointer_writer=True,
            candidate_real_capture_params=False,direct_params_driver_metadata=True,
            candidate_pointer_order=['packed_filter_high','packed_filter_low','native_input_A','native_output_C'],
            cached_native_filter_bytes=numel*4,packed_filter_bytes=numel*4)

    def refresh_if_needed(self):
        assert not self.closed and not self.cache.enabled
        self.cache.refresh_if_needed()
        if self.packed_version!=self.cache.version:
            n=self.cache.cache.numel()
            pack_native_filter[(triton.cdiv(n,1024),)](self.cache.cache,self.wh,self.wl,n,1024,
                num_warps=4,enable_fp_fusion=False)
            self.wh.record_stream(torch.cuda.current_stream(self.wh.device))
            self.wl.record_stream(torch.cuda.current_stream(self.wl.device))
            self.packed_version=self.cache.version;self.refreshes+=1

    def set_enabled(self,on):
        assert not self.closed and not torch.cuda.is_current_stream_capturing()
        if on:self.refresh_if_needed()
        params=self.replacement if on else self.original
        try:
            self.api.check(self.api.lib.cuGraphExecKernelNodeSetParams(self.executable,self.node,
                C.byref(params.params)),'set packed GEMM parameters')
            self.api.check(self.api.lib.cuGraphNodeSetEnabled(self.executable,self.transform_node,0 if on else 1),
                'toggle native filter under packed owner')
            value=C.c_uint()
            self.api.check(self.api.lib.cuGraphNodeGetEnabled(self.executable,self.transform_node,C.byref(value)),
                'verify packed native filter toggle')
            assert bool(value.value)==(not on)
        except Exception:
            self.api.lib.cuGraphExecKernelNodeSetParams(self.executable,self.node,C.byref(self.original.params))
            self.api.lib.cuGraphNodeSetEnabled(self.executable,self.transform_node,1)
            self.enabled=False
            raise
        self.enabled=bool(on)

    def verify_cache(self):
        self.cache.verify_cache()
        high=self.cache.cache.half()
        low=((self.cache.cache-high.float())*2048.).half()
        assert torch.equal(self.wh.view(torch.int16),high.view(torch.int16))
        assert torch.equal(self.wl.view(torch.int16),low.view(torch.int16))
        for buf in [self.wh_backing,self.wl_backing]:
            assert torch.all(buf[:self.guard]==123.5) and torch.all(buf[-self.guard:]==123.5)

    def poison_old_workspace(self):
        assert self.enabled
        self.api.check(self.api.lib.cuMemsetD32Async(self.cache.old_output,0x7fc00000,
            self.cache.cache.numel(),C.c_void_p(torch.cuda.current_stream(self.wh.device).cuda_stream)),
            'poison original filter workspace under packed GEMM')

    def replay(self):
        assert not self.closed
        if self.enabled:self.refresh_if_needed()
        self.graph.replay()

    def close(self):
        if self.closed:return
        torch.cuda.synchronize()
        self.set_enabled(False)
        self.cache.close()
        self.closed=True


class NativeOnlyFilterSwap:
    """Cache an omitted convolution's original native filter; keep FP32 SGEMM."""
    def __init__(self,graph,weight,plan,extension,api):
        self.cache=NativeWorldFilterCache(graph,weight,extension,api,validated_plan=plan)
        self.proof=dict(original_native_input_filter_output_transforms=True,
            original_filter_cloned_on_master_version=True,numerical_SGEMM_only=False,
            native_FP32_SGEMM_unchanged=True,single_executable_pointer_writer=True,
            cached_native_filter_bytes=self.cache.cache.numel()*4)

    def refresh_if_needed(self):self.cache.refresh_if_needed()
    def set_enabled(self,on):self.cache.set_cached(on)
    def verify_cache(self):self.cache.verify_cache()
    def close(self):self.cache.close()
