"""Cache native Conv1d filter transforms; preserve original graph ownership."""
import json,os,struct
import torch
from unifolm_wma.utils.native_graph_weight_cache_probe import TRANSPOSE,XMMA,XMMA_NONINDEXED,NativeWeightCache
CUTLASS_64='_ZN19cutlass_cudnn_infer6KernelI65cutlass_tensorop_s1688fprop_optimized_tf32_64x64_32x5_nhwc_align4EEvNT_6ParamsE'
def plan_for(nodes,outgoing,transform,weight):
    assert weight.is_contiguous() and weight.dtype==torch.float32 and weight.ndim==3
    matches=[v for v in SCHEMAS.values() if v[0]==tuple(weight.shape)]
    assert matches
    co,ci,spatial=weight.shape
    args=transform['arguments'];assert [a['size'] for a in args]==[44,8,8]
    descriptor=struct.unpack('<11I',bytes.fromhex(args[0]['bytes_hex']))
    assert descriptor==(co,ci,ci,ci*spatial,spatial,spatial,ci,ci*spatial,ci,0x3f800000,0)
    assert int.from_bytes(bytes.fromhex(args[1]['bytes_hex']),'little')==weight.data_ptr()
    old=int.from_bytes(bytes.fromhex(args[2]['bytes_hex']),'little')
    frontier=outgoing[transform['node']];path=[]
    for _ in range(8):
        assert len(frontier)==1
        consumer=nodes[frontier[0]]
        if consumer['type']==2:pass
        elif consumer.get('name')==TRANSPOSE:
            assert [a['size'] for a in consumer['arguments']]==[44,8,8]
            assert consumer['node']!=transform['node']
        else:break
        path.append(consumer['node']);frontier=outgoing[consumer['node']]
    else:raise AssertionError('native dependency chain exceeds audited bound')
    matches=[v for v in matches if v[1]==descriptor and v[2]==consumer['name']]
    assert matches and len(consumer.get('arguments',[]))==1
    assert len({(v[3],v[4]) for v in matches})==1
    size,byte=matches[0][3:]
    payload=bytes.fromhex(consumer['arguments'][0]['bytes_hex'])
    assert len(payload)==size and int.from_bytes(payload[byte:byte+8],'little')==old
    return transform,consumer,byte,old,path


FLAG='UNIFOLM_WORLD_HEAD_FILTER_CACHE'
DEPENDENCIES={'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE': '1', 'UNIFOLM_DENOISER_MODE': 'native', 'UNIFOLM_MATH_MODE': 'fp32', 'UNIFOLM_NATIVE_SGEMM_FP16': '1', 'UNIFOLM_SIM_HEAD_TF32': '1', 'UNIFOLM_NATIVE_REMAINING_FILTER_CACHE': '1'}


def configured(environment):
    raw=environment.get(FLAG,'0')
    if raw not in ('0','1'):raise ValueError('Invalid native Conv1d cache flag')
    if raw=='0' or environment.get('UNIFOLM_CASE')!='unitree_z1_dual_arm_stackbox_v2':return False
    for name,value in DEPENDENCIES.items():
        if environment.get(name)!=value:raise ValueError('Native Conv1d cache requires '+name+'='+value)
    return True


class Controller:
    def __init__(self,owner,model,extension,api,inventory):
        assert not torch.is_grad_enabled() and not torch.cuda.is_current_stream_capturing()
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        assert len(owner.swaps)==23 and owner.report['selected']==23
        assert not hasattr(owner,'_world_head_filter_cache')
        self.owner,self.caches,self.closed=owner,[],False
        self.original_replay,self.original_close=owner.replay,owner.close
        self.old_swaps=tuple(owner.swaps)
        self.stream=torch.cuda.current_stream()
        nodes={n['node']:n for n in inventory['nodes']};outgoing={}
        for a,b in inventory['edges']:outgoing.setdefault(a,[]).append(b)
        masters={p.data_ptr():(name,p) for name,p in model.named_parameters() if name in SCHEMAS}
        assert len(masters)==45
        plans=[]
        for n in nodes.values():
            if n.get('name')!=TRANSPOSE:continue
            ptr=int.from_bytes(bytes.fromhex(n['arguments'][1]['bytes_hex']),'little')
            if ptr not in masters:continue
            name,weight=masters[ptr]
            plan=plan_for(nodes,outgoing,n,weight)
            shape,descriptor,consumer,size,byte=SCHEMAS[name]
            assert tuple(weight.shape)==shape and tuple(struct.unpack('<11I',bytes.fromhex(n['arguments'][0]['bytes_hex'])))==descriptor
            assert plan[1]['name']==consumer and plan[2]==byte and plan[1]['arguments'][0]['size']==size
            plans.append((name,weight,plan))
        assert len(plans)==45 and len({p.data_ptr() for _,p,_ in plans})==45
        assert not any(plan[1]['node']==hex(c.consumer_node) for _,_,plan in plans for c in (s.cache for s in owner.swaps))
        try:
            for name,weight,plan in plans:
                self.caches.append((name,NativeWeightCache(owner.graph,weight,extension,api,validated_plan=plan)))
            self.set_enabled(True)
        except Exception:
            for _,c in self.caches:c.close()
            raise
        owner._world_head_filter_cache=self
        owner.replay=self.replay;owner.close=self.close
        print('WORLD_HEAD_FILTER_CACHE '+json.dumps(dict(selected=45,
            cached_bytes=sum(c.cache.numel()*4 for _,c in self.caches),original_numerical_swaps=23,
            original_native_transform_and_convolution=True,live_weight_version_refresh=True,
            no_activation_cache=True,all_setup_inside_process_wall=True)),flush=True)

    def set_enabled(self,enabled):
        assert not self.closed
        for _,c in self.caches:c.set_cached(enabled)
        self.enabled=enabled

    def replay(self):
        assert not self.closed and tuple(self.owner.swaps)==self.old_swaps
        if self.enabled:
            current=torch.cuda.current_stream()
            for _,c in self.caches:
                if current!=self.stream:c.cache.record_stream(current)
                c.refresh_if_needed()
            self.stream=current
        return self.original_replay()

    def close(self):
        if self.closed:return
        for _,c in self.caches:c.close()
        self.owner.replay=self.original_replay;self.owner.close=self.original_close
        self.closed=True
        self.original_close()


def attach(owner,model,extension,api,inventory):
    if not configured(os.environ):return None
    return Controller(owner,model,extension,api,inventory)

SCHEMAS={'model.diffusion_model.state_unet.down_modules.0.0.blocks.0.block.0.weight': ((256, 16, 5), (256, 16, 16, 80, 5, 5, 16, 80, 16, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn_infer', 1440, 8), 'model.diffusion_model.state_unet.down_modules.0.0.blocks.1.block.0.weight': ((256, 256, 5), (256, 256, 256, 1280, 5, 5, 256, 1280, 256, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn_infer', 1440, 8), 'model.diffusion_model.state_unet.down_modules.0.1.blocks.0.block.0.weight': ((256, 256, 5), (256, 256, 256, 1280, 5, 5, 256, 1280, 256, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn_infer', 1440, 8), 'model.diffusion_model.state_unet.down_modules.0.1.blocks.1.block.0.weight': ((256, 256, 5), (256, 256, 256, 1280, 5, 5, 256, 1280, 256, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn_infer', 1440, 8), 'model.diffusion_model.state_unet.down_modules.0.3.conv.weight': ((256, 256, 3), (256, 256, 256, 768, 3, 3, 256, 768, 256, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn_infer', 1440, 8), 'model.diffusion_model.state_unet.down_modules.1.0.blocks.0.block.0.weight': ((512, 256, 5), (512, 256, 256, 1280, 5, 5, 256, 1280, 256, 1065353216, 0), '_ZN19cutlass_cudnn_infer6KernelI65cutlass_tensorop_s1688fprop_optimized_tf32_64x64_32x5_nhwc_align4EEvNT_6ParamsE', 464, 240), 'model.diffusion_model.state_unet.down_modules.1.0.blocks.1.block.0.weight': ((512, 512, 5), (512, 512, 512, 2560, 5, 5, 512, 2560, 512, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.1.1.blocks.0.block.0.weight': ((512, 512, 5), (512, 512, 512, 2560, 5, 5, 512, 2560, 512, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.1.1.blocks.1.block.0.weight': ((512, 512, 5), (512, 512, 512, 2560, 5, 5, 512, 2560, 512, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.1.3.conv.weight': ((512, 512, 3), (512, 512, 512, 1536, 3, 3, 512, 1536, 512, 1065353216, 0), '_ZN19cutlass_cudnn_infer6KernelI65cutlass_tensorop_s1688fprop_optimized_tf32_64x64_32x5_nhwc_align4EEvNT_6ParamsE', 464, 240), 'model.diffusion_model.state_unet.down_modules.2.0.blocks.0.block.0.weight': ((1024, 512, 5), (1024, 512, 512, 2560, 5, 5, 512, 2560, 512, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.2.0.blocks.1.block.0.weight': ((1024, 1024, 5), (1024, 1024, 1024, 5120, 5, 5, 1024, 5120, 1024, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.2.1.blocks.0.block.0.weight': ((1024, 1024, 5), (1024, 1024, 1024, 5120, 5, 5, 1024, 5120, 1024, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.2.1.blocks.1.block.0.weight': ((1024, 1024, 5), (1024, 1024, 1024, 5120, 5, 5, 1024, 5120, 1024, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.2.3.conv.weight': ((1024, 1024, 3), (1024, 1024, 1024, 3072, 3, 3, 1024, 3072, 1024, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 1440, 8), 'model.diffusion_model.state_unet.down_modules.3.0.blocks.0.block.0.weight': ((2048, 1024, 5), (2048, 1024, 1024, 5120, 5, 5, 1024, 5120, 1024, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.3.0.blocks.1.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.3.1.blocks.0.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.down_modules.3.1.blocks.1.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.mid_modules.0.blocks.0.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.mid_modules.0.blocks.1.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.mid_modules.1.blocks.0.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.mid_modules.1.blocks.1.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.0.0.blocks.0.block.0.weight': ((2048, 4096, 5), (2048, 4096, 4096, 20480, 5, 5, 4096, 20480, 4096, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.0.0.blocks.1.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.0.0.residual_conv.weight': ((2048, 4096, 1), (2048, 4096, 4096, 4096, 1, 1, 4096, 4096, 4096, 1065353216, 0), '_ZN19cutlass_cudnn_infer6KernelI65cutlass_tensorop_s1688fprop_optimized_tf32_64x64_32x5_nhwc_align4EEvNT_6ParamsE', 464, 240), 'model.diffusion_model.state_unet.up_modules.0.1.blocks.0.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.0.1.blocks.1.block.0.weight': ((2048, 2048, 5), (2048, 2048, 2048, 10240, 5, 5, 2048, 10240, 2048, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.0.3.conv.weight': ((2048, 2048, 4), (2048, 2048, 2048, 8192, 4, 4, 2048, 8192, 2048, 1065353216, 0), 'sm80_xmma_dgrad_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 1456, 8), 'model.diffusion_model.state_unet.up_modules.1.0.blocks.0.block.0.weight': ((1024, 3072, 5), (1024, 3072, 3072, 15360, 5, 5, 3072, 15360, 3072, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.1.0.blocks.1.block.0.weight': ((1024, 1024, 5), (1024, 1024, 1024, 5120, 5, 5, 1024, 5120, 1024, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.1.0.residual_conv.weight': ((1024, 3072, 1), (1024, 3072, 3072, 3072, 1, 1, 3072, 3072, 3072, 1065353216, 0), '_ZN19cutlass_cudnn_infer6KernelI65cutlass_tensorop_s1688fprop_optimized_tf32_64x64_32x5_nhwc_align4EEvNT_6ParamsE', 464, 240), 'model.diffusion_model.state_unet.up_modules.1.1.blocks.0.block.0.weight': ((1024, 1024, 5), (1024, 1024, 1024, 5120, 5, 5, 1024, 5120, 1024, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.1.1.blocks.1.block.0.weight': ((1024, 1024, 5), (1024, 1024, 1024, 5120, 5, 5, 1024, 5120, 1024, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.1.3.conv.weight': ((1024, 1024, 4), (1024, 1024, 1024, 4096, 4, 4, 1024, 4096, 1024, 1065353216, 0), 'sm80_xmma_dgrad_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 1456, 8), 'model.diffusion_model.state_unet.up_modules.2.0.blocks.0.block.0.weight': ((512, 1536, 5), (512, 1536, 1536, 7680, 5, 5, 1536, 7680, 1536, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.2.0.blocks.1.block.0.weight': ((512, 512, 5), (512, 512, 512, 2560, 5, 5, 512, 2560, 512, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.2.1.blocks.0.block.0.weight': ((512, 512, 5), (512, 512, 512, 2560, 5, 5, 512, 2560, 512, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.2.1.blocks.1.block.0.weight': ((512, 512, 5), (512, 512, 512, 2560, 5, 5, 512, 2560, 512, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.2.3.conv.weight': ((512, 512, 4), (512, 512, 512, 2048, 4, 4, 512, 2048, 512, 1065353216, 0), '_ZN19cutlass_cudnn_infer6KernelI66cutlass_tensorop_s1688dgrad_optimized_tf32_64x128_32x3_nhwc_align4EEvNT_6ParamsE', 624, 240), 'model.diffusion_model.state_unet.up_modules.3.0.blocks.0.block.0.weight': ((256, 768, 5), (256, 768, 768, 3840, 5, 5, 768, 3840, 768, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer', 784, 8), 'model.diffusion_model.state_unet.up_modules.3.0.blocks.1.block.0.weight': ((256, 256, 5), (256, 256, 256, 1280, 5, 5, 256, 1280, 256, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn_infer', 1440, 8), 'model.diffusion_model.state_unet.up_modules.3.1.blocks.0.block.0.weight': ((256, 256, 5), (256, 256, 256, 1280, 5, 5, 256, 1280, 256, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn_infer', 1440, 8), 'model.diffusion_model.state_unet.up_modules.3.1.blocks.1.block.0.weight': ((256, 256, 5), (256, 256, 256, 1280, 5, 5, 256, 1280, 256, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn_infer', 1440, 8), 'model.diffusion_model.state_unet.final_conv.0.block.0.weight': ((256, 256, 5), (256, 256, 256, 1280, 5, 5, 256, 1280, 256, 1065353216, 0), 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel_cudnn_infer', 1440, 8)}
