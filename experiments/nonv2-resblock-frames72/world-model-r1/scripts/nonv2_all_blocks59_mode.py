"""Share the existing33 gathers across full inner Half Transformer blocks."""
from contextlib import contextmanager
import torch
import torch.distributed as dist
import nonv2_ffn59_mode as base
import nonv2_temporal_block59_mode as temporal

active=None


def sample_slice(value,rank):
    assert value.shape[0]==16 and rank in (0,1)
    return value[8*rank:8*(rank+1)]


def action_bias_slice(value,rank,heads):
    assert value.ndim==3 and value.shape[0]==16*heads
    return value[8*rank*heads:8*(rank+1)*heads]


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import SpatialTransformer,BasicTransformerBlock
        from unifolm_wma.utils.triton_attention_mix import _mix
        self.inner=temporal.install(model);self.saved=self.inner.saved;self.rows=self.inner.rows
        self.rank=self.inner.rank;self.blocks=[];self.hooks=[];self.caches=[];self.inside_spatial=False
        originals={id(m):fn for m,fn in self.saved}
        for parent_name,parent in model.named_modules():
            if type(parent)is not SpatialTransformer:continue
            for index,block in enumerate(parent.transformer_blocks):
                assert type(block)is BasicTransformerBlock
                name=parent_name+'.transformer_blocks.'+str(index)
                ff=originals[id(block.ff)]
                def local_ffn(value,_forward=ff):
                    if not base.enabled:return _forward(value)
                    with base.candidate_cache(self.rank):return _forward(value)
                block.ff.forward=local_ffn
                for attention in (block.attn1,block.attn2):
                    if not attention.image_cross_attention:continue
                    self.caches.append((attention,attention._condition_projection_cache,{}))
                    before=attention.prepare_condition_cache
                    def prepare(context,_before=before):
                        return _before(sample_slice(context,self.rank)if base.enabled else context)
                    self.hooks.append((attention,'prepare_condition_cache',before));attention.prepare_condition_cache=prepare
                    before=attention._get_cached_attn_mask_aa
                    def bias(b,l1,l2,*,device,dtype,block_size=16,_before=before,_heads=attention.heads):
                        if not base.enabled:return _before(b,l1,l2,device=device,dtype=dtype,block_size=block_size)
                        assert b==8 and block_size==16
                        whole=_before(16,l1,l2,device=device,dtype=dtype,block_size=block_size)
                        return action_bias_slice(whole,self.rank,_heads)
                    self.hooks.append((attention,'_get_cached_attn_mask_aa',before));attention._get_cached_attn_mask_aa=bias
                before=block.forward
                def forward(value,context=None,mask=None,_before=before,_name=name,_block=block,**kwargs):
                    if not base.enabled:return _before(value,context=context,mask=mask,**kwargs)
                    assert not kwargs and mask is None and context is not None
                    assert value.is_cuda and value.dtype==torch.float16 and value.is_contiguous()
                    assert not torch.is_grad_enabled()and not torch.is_autocast_enabled()and not self.inside_spatial
                    assert all(not m.training for m in _block.modules())
                    local=sample_slice(value,self.rank);condition=sample_slice(context,self.rank)
                    self.inside_spatial=True
                    try:result=_before(local,context=condition,mask=None)
                    finally:self.inside_spatial=False
                    assert result.shape==local.shape and result.dtype==value.dtype
                    output=torch.empty_like(value,memory_format=torch.contiguous_format)
                    dist.all_gather_into_tensor(output,result.contiguous())
                    if torch.cuda.is_current_stream_capturing():
                        self.rows.append(dict(module=_name,input_shape=list(value.shape),rank=self.rank,collectives=1,
                            local_output_MiB=output.numel()*output.element_size()/2**21,scope='spatial_inner_block',
                            original_global_frame_action_mask=True,condition_cache_banked_per_mode=True,
                            original_block_forward_and_Half_master_storage=True,additional_collectives=0))
                    return output
                block.forward=forward;self.blocks.append((block,before))
        assert len(self.blocks)==16 and len(self.caches)==16
        # Reuse the unmodified warm mix specialization. Each selected extent is
        # a whole1024-element tile; the original half-size launch grid visits
        # exactly the local elements. Its original full-size bound is therefore
        # inactive on every launched lane. No pointer or arithmetic is changed.
        before=_mix.run
        def run(*args,**kwargs):
            if base.enabled and self.inside_spatial:
                values=list(args);n=values[5]
                assert isinstance(n,int)and n>0 and n%1024==0 and kwargs['grid']==(n//1024,)
                values[5]=2*n;args=tuple(values)
            return before(*args,**kwargs)
        _mix.run=run;self.hooks.append((_mix,'run',before))

    def masters(self):
        return {p.data_ptr():p for module in [*(m for m,_ in self.saved),*(m for m,_ in self.inner.blocks),*(m for m,_ in self.blocks)]for p in module.parameters()}

    def report(self):
        report=self.inner.report();report.update(spatial_blocks=16,spatial_block_captured=sum(r.get('scope')=='spatial_inner_block'for r in self.rows),
            spatial_FFNs=0,same33_collectives_as_FFN_only=True,global_action_mask_preserved=True,condition_caches_banked=True,
            mix_uses_original_warm_specialization_with_whole_local_tile_grid=True)
        return report


def install(model):
    global active
    assert active is None;active=Controller(model);return active


def uninstall():
    global active
    if active is not None:
        for obj,name,value in reversed(active.hooks):setattr(obj,name,value)
        for module,forward in reversed(active.blocks):module.forward=forward
        for attention,baseline,_ in active.caches:attention._condition_projection_cache=baseline
    temporal.uninstall();active=None


@contextmanager
def mode(value):
    assert active is not None
    previous=[(m,m._condition_projection_cache)for m,_,_ in active.caches]
    for module,baseline,candidate in active.caches:module._condition_projection_cache=candidate if value else baseline
    try:
        with base.mode(value):yield
    finally:
        for module,cache in previous:module._condition_projection_cache=cache
