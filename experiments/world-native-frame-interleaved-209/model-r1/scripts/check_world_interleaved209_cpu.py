"""Check spatial frame routing and full temporal boundaries without a GPU."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.networks.wma_model import ResBlock
from unifolm_wma.utils import triton_group_norm as normalization
import nccl_world_interleaved209_mode as candidate




def check_world_owners():
    from pathlib import Path
    scope=json.loads((Path(__file__).resolve().parent.parent/'world-interleaved-owner-scope.json').read_text())
    names={};geometry={};swaps=[];plans=[]
    for row in scope['rows']:
        co,cols,ci=row['geometry'];h,w={2560:(40,64),640:(20,32),192:(10,16)}[cols]
        weight=torch.empty((co,ci,3,3),device='meta')
        swaps.append(SimpleNamespace(cache=SimpleNamespace(weight=weight)))
        plans.append(dict(weight='extra.prefix.'+row['weight'],geometry=row['geometry']))
        if row['within_qualified_ResBlock']:
            local=row['weight'].removeprefix('model.diffusion_model.')
            names[id(weight)]=local;geometry[local]=dict(co=co,ci=ci,height=h,width=w,full_frames=16)
    owner=SimpleNamespace(swaps=swaps,report=dict(selected=23,numerical_selected=15,kahan_selected=8,plans=plans),
        _world_head_filter_cache=SimpleNamespace(caches=[None]*45))
    eligible,outside=candidate.bind_owners(owner,names,geometry)
    assert len(eligible)==21 and len(outside)==2
    assert len({n.rsplit('.',3)[0] for n,_,_ in eligible})==12
    assert not any(g in candidate.SLOW for _,_,g in eligible)
    print('WORLD_OWNER_IDENTITY_BINDING_CPU passed=true owners=21 blocks=12 excluded_upsample=2')


def norm_cpu(value,norm,*,scale=None,shift=None,apply_silu=True):
    result=norm(value)
    if scale is not None:result=result*(1+scale)+shift
    return torch.nn.functional.silu(result) if apply_silu else result


def main():
    torch.manual_seed(311);assert not torch.cuda.is_initialized()
    old=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
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
                                        controller=candidate.install(torch.nn.ModuleList([block]),selection=dict(names=['0'],expected_total=1))
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
    check_world_owners()
    assert not torch.cuda.is_initialized()
    print('NCCL_WORLD_INTERLEAVED_CPU_CHECK '+json.dumps(dict(passed=True,cases=len(records),
        all_ranks_and_three_layouts=True,full_embedding_rows_preserved=True,
        original_full16_temporal_boundaries=True,no_block_exclusion_rules=True,original_N16_descriptor_shapes=True,
        world_owner_identity_binding_with_extra_prefix=True,
        all_hooks_restored=True,no_GPU_initialized=True)),flush=True)


if __name__=='__main__':main()
