"""Check spatial frame routing and full temporal boundaries without a GPU."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.networks.wma_model import ResBlock
from unifolm_wma.utils import triton_group_norm as normalization
import nccl_resblock_native_grid209_mode as candidate


def check_cached_pointer_owner_hook():
    from unifolm_wma.utils.native_graph_weight_cache import NativeGraphWeights
    from unifolm_wma.utils.native_graph_weight_cache_probe import CUTLASS, OwnedParams
    block=ResBlock(32,128,0.).eval()
    model=torch.nn.ModuleList([block])
    caches=[]
    originals={}
    for suffix in ('in_layers.2','out_layers.3'):
        name='0.'+suffix+'.weight'
        data=bytearray(464);data[240:248]=(1).to_bytes(8,'little')
        node=dict(function='0x1',grid=[3,1,1],block=[128,1,1],shared_bytes=0,
            arguments=[dict(index=0,bytes_hex=data.hex())])
        original=OwnedParams(node)
        cached=OwnedParams(node,{(0,240):(1,2)})
        cache=SimpleNamespace(weight=torch.empty(32,32,3,3),original_consumer=original,
            cached_consumer=cached,proof=dict(consumer=CUTLASS),calls=[])
        def select(value,_cache=cache):
            _cache.calls.append(bool(value))
        cache.set_cached=select
        caches.append((name,cache))
        originals[name]=(bytes(original.buffers[0]),bytes(cached.buffers[0]))
    old=NativeGraphWeights.__init__
    def initialize(owner,graph,model):
        owner.caches=caches;owner.report=dict(selected=138)
    with patch.object(NativeGraphWeights,'__init__',initialize),patch.object(candidate.dist,'get_world_size',return_value=2),patch.object(candidate.dist,'get_rank',return_value=0):
        controller=candidate.install(model)
        try:
            for name,_ in caches:
                controller.geometry[name]=dict(co=32,ci=32,height=4,width=6,
                    input_stride=[768,24,6,1],full_frames=16)
            owner=SimpleNamespace()
            with candidate.mode(True):controller.initialize(owner,None,model)
            assert len(controller.native_reports)==1
            for name,cache in caches:
                assert cache.calls==[True]
                assert bytes(cache.original_consumer.buffers[0])==originals[name][0]
                assert bytes(cache.cached_consumer.buffers[0])==originals[name][1]
                assert int.from_bytes(bytes(cache.cached_consumer.buffers[0])[240:248],'little')==2
                assert cache.cached_consumer.params.grid_x==2
                assert cache.original_consumer.params.grid_x==3
            assert controller.native_reports[0]['selected']==2
        finally:candidate.uninstall()
    assert NativeGraphWeights.__init__==old


def norm_cpu(value,norm,*,scale=None,shift=None,apply_silu=True):
    result=norm(value)
    if scale is not None:result=result*(1+scale)+shift
    return torch.nn.functional.silu(result) if apply_silu else result


def main():
    torch.manual_seed(311);assert not torch.cuda.is_initialized()
    old=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    try:
        configurations=[{},dict(out_channels=64),dict(out_channels=64,use_conv=True),
            dict(use_scale_shift_norm=True),dict(up=True),dict(down=True),
            dict(use_temporal_conv=True),dict(use_temporal_conv=True,use_scale_shift_norm=True)]
        for config in configurations:
            block=ResBlock(32,128,0.,**config).eval()
            with torch.no_grad():
                for p in block.parameters():
                    if p.ndim>=2:p.normal_(0,.02)
            for layout in ('nchw','nc_permuted','channels_last'):
                value=torch.randn(16,32,4,6)
                if layout=='nc_permuted':value=value.transpose(0,1).contiguous().transpose(0,1)
                if layout=='channels_last':value=value.contiguous(memory_format=torch.channels_last)
                for emb_rows in (1,16):
                    emb=torch.randn(emb_rows,128)
                    for batch_size in ((None,1) if config.get('use_temporal_conv') else (1,)):
                        spatial=[];embedding=[];temporal=[]
                        def capture_temporal(module,args):
                            tensor=args[0];assert tensor.shape[:3]==(1,block.out_channels,16)
                            spatial.append(tensor.permute(0,2,1,3,4).reshape(16,block.out_channels,*tensor.shape[-2:]).clone())
                            temporal.append(tuple(tensor.shape))
                        hook=block.temopral_conv.register_forward_pre_hook(capture_temporal) if config.get('use_temporal_conv') and batch_size else None
                        emb_hook=block.emb_layers[-1].register_forward_pre_hook(lambda module,args:embedding.append(args[0].shape[0]))
                        try:
                            with torch.no_grad(),patch.object(normalization,'fused_group_norm',side_effect=norm_cpu):
                                # Include the production fused-norm branch in half the fixtures.
                                block._fused_groupnorm_silu=bool(emb_rows==16)
                                expected=block(value,emb,batch_size=batch_size)
                                expected_spatial=spatial[0] if spatial else expected.clone()
                                for rank in (0,1):
                                    calls=[];original=block._forward
                                    def gather(output,local):
                                        reference=expected_spatial[rank*8:(rank+1)*8]
                                        assert local.shape[0]==8
                                        torch.testing.assert_close(local,reference,rtol=2e-5,atol=2e-6)
                                        assert output.shape==expected_spatial.shape
                                        output.copy_(expected_spatial);calls.append(tuple(local.shape))
                                    with patch.object(candidate.dist,'get_world_size',return_value=2),patch.object(candidate.dist,'get_rank',return_value=rank),patch.object(candidate.dist,'all_gather_into_tensor',side_effect=gather),patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
                                        controller=candidate.install(torch.nn.ModuleList([block]))
                                        try:
                                            with candidate.mode(True):actual=block(value,emb,batch_size=batch_size)
                                            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
                                            assert len(calls)==1 and controller.current is None
                                            assert controller.report()['ResBlocks']==1
                                        finally:candidate.uninstall()
                                    assert block._forward==original
                                    records.append(dict(config=config,layout=layout,embedding_rows=emb_rows,
                                        rank=rank,batch_size=batch_size,full16_temporal=bool(spatial)))
                                assert embedding==[emb_rows]*3,embedding
                                if spatial:assert len(temporal)==3
                        finally:
                            emb_hook.remove()
                            if hook is not None:hook.remove()
    finally:
        candidate.uninstall();torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32=old
    check_cached_pointer_owner_hook()
    assert not torch.cuda.is_initialized()
    print('NCCL_RESBLOCK_NATIVE_GRID_CPU_CHECK '+json.dumps(dict(passed=True,cases=len(records),
        all_ranks_and_three_layouts=True,full_embedding_rows_preserved=True,
        original_full16_temporal_boundaries=True,no_block_exclusion_rules=True,original_N16_descriptor_shapes=True,
        cached_filter_pointer_and_original_close_grid_preserved=True,
        all_hooks_restored=True,no_GPU_initialized=True)),flush=True)


if __name__=='__main__':main()
