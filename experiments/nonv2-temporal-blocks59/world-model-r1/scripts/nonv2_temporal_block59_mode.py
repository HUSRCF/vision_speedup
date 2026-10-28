"""Expand17 temporal FFN gathers to whole inner blocks on nonV2 Half math."""
import torch
import torch.distributed as dist
import nonv2_ffn59_mode as base
from nonv2_ffn59_mode import mode

active=None


def temporal_inference(value,rank,forward,gather,mask=None):
    assert rank in (0,1)and value.ndim==3 and value.is_contiguous()
    assert value.shape[0]>0 and value.shape[0]%2==0 and value.shape[1]==16
    half=value.shape[0]//2;rows=slice(rank*half,(rank+1)*half)
    if mask is not None:
        assert mask.ndim==3 and mask.shape==(value.shape[0],16,16)
        local_mask=mask[rows]
    else:local_mask=None
    local=forward(value[rows],context=None,mask=local_mask)
    assert local.shape==(half,*value.shape[1:])and local.dtype==value.dtype and local.device==value.device
    output=torch.empty_like(value,memory_format=torch.contiguous_format)
    gather(output,local.contiguous())
    return output


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import TemporalTransformer,BasicTransformerBlock
        self.inner=base.install(model);self.saved=self.inner.saved;self.rows=self.inner.rows
        originals={id(module):forward for module,forward in self.saved}
        self.blocks=[];self.rank=self.inner.rank
        for parent_name,parent in model.named_modules():
            if type(parent)is not TemporalTransformer:continue
            assert parent.only_self_att and not parent.causal_attention
            for index,block in enumerate(parent.transformer_blocks):
                assert type(block)is BasicTransformerBlock
                name=parent_name+'.transformer_blocks.'+str(index)
                old_ffn=originals[id(block.ff)]
                def local_ffn(value,_forward=old_ffn):
                    if not base.enabled:return _forward(value)
                    # Reuse the already-qualified half-M FFN shape cache only.
                    with base.candidate_cache(self.rank):return _forward(value)
                block.ff.forward=local_ffn
                before=block.forward
                def forward(value,context=None,mask=None,_before=before,_name=name,_block=block,**kwargs):
                    if not base.enabled:return _before(value,context=context,mask=mask,**kwargs)
                    assert not kwargs and context is None
                    assert value.is_cuda and value.dtype==torch.float16
                    assert not torch.is_grad_enabled()and not torch.is_autocast_enabled()
                    assert all(not m.training for m in _block.modules())
                    assert all(p.dtype==torch.float16 and p.device==value.device for p in _block.parameters())
                    result=temporal_inference(value,self.rank,_before,dist.all_gather_into_tensor,mask)
                    if torch.cuda.is_current_stream_capturing():
                        self.rows.append(dict(module=_name,input_shape=list(value.shape),dtype=str(value.dtype),rank=self.rank,collectives=1,
                            local_output_MiB=result.numel()*result.element_size()/2**21,
                            scope='temporal_inner_block',full16_time_steps=True,full5D_GroupNorm_outside_shard=True,
                            original_block_forward_and_Half_master_storage=True,additional_collectives=0))
                    return result
                block.forward=forward;self.blocks.append((block,before))
        assert len(self.blocks)==17 and len(self.saved)==33

    def masters(self):
        return {p.data_ptr():p for module in [*(m for m,_ in self.saved),*(m for m,_ in self.blocks)]for p in module.parameters()}

    def report(self):
        report=self.inner.report()
        report.update(temporal_blocks=17,spatial_FFNs=16,temporal_block_captured=sum(r.get('scope')=='temporal_inner_block'for r in self.rows),
                      same33_collectives_as_FFN_only=True,full5D_GroupNorm_and_boundary_projections_outside_shard=True,
                      temporal_QKV_attention_norm_residual_FFN_sharded=True,baseline_is_original59_nonv2_FP16=True)
        return report


def install(model):
    global active
    assert active is None;active=Controller(model);return active


def uninstall():
    global active
    if active is not None:
        for module,forward in reversed(active.blocks):module.forward=forward
    base.uninstall();active=None
