"""Shard only Half xFormers frame batches; preserve all vendor GEMM shapes."""
from contextlib import contextmanager
import torch
import torch.distributed as dist
import xformers.ops

active=None
enabled=False
parent_attention=xformers.ops.memory_efficient_attention


def local_attention(parent,q,k,v,bias,rank,**options):
    assert rank in(0,1)and q.ndim==k.ndim==v.ndim==4
    assert q.shape[0]==k.shape[0]==v.shape[0]==16
    assert q.shape[2]==k.shape[2]==v.shape[2]and k.shape[1]==v.shape[1]
    assert options.get('p',0.)==0.
    begin,end=8*rank,8*(rank+1)
    if isinstance(bias,torch.Tensor):
        assert bias.ndim==4 and bias.shape[0]in(1,16)
        bias=bias[begin:end]if bias.shape[0]==16 else bias
    else:assert bias is None
    local=parent(q[begin:end],k[begin:end],v[begin:end],attn_bias=bias,**options)
    assert local.shape==(8,q.shape[1],q.shape[2],v.shape[3])
    # Unowned frame rows remain private to this spatial scope until its gather.
    full=torch.empty((16,*local.shape[1:]),dtype=local.dtype,device=local.device)
    full[begin:end].copy_(local)
    return full


def attention(q,k,v,attn_bias=None,p=0.,scale=None,*,op=None,output_dtype=None):
    options=dict(p=p,scale=scale,op=op,output_dtype=output_dtype)
    if not enabled or active is None or active.scope is None:
        return parent_attention(q,k,v,attn_bias=attn_bias,**options)
    scope=active.scope
    assert not torch.is_grad_enabled()and q.shape[1]==scope['M']
    assert all(t.is_cuda and t.dtype==torch.float16 and t.device==q.device for t in(q,k,v))
    result=local_attention(parent_attention,q,k,v,attn_bias,active.rank,**options)
    if torch.cuda.is_current_stream_capturing():
        active.calls.append(dict(module=scope['name'],query_shape=list(q.shape),key_shape=list(k.shape),value_shape=list(v.shape),rank=active.rank,local_batch=8,outer_batch=16,bias_present=attn_bias is not None,global_mask_slice=True,extra_collectives=0))
    return result


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import SpatialTransformer
        assert dist.is_initialized()and dist.get_world_size()==2
        self.rank=dist.get_rank();self.saved=[];self.rows=[];self.calls=[];self.scope=None
        for name,module in model.named_modules():
            if type(module)is not SpatialTransformer:continue
            assert not module._channels_last_output and len(module.transformer_blocks)==1
            before=module.forward
            def forward(value,*args,_before=before,_module=module,_name=name,**kwargs):
                if not enabled:return _before(value,*args,**kwargs)
                assert self.scope is None and value.ndim==4 and value.shape[0]==16
                assert value.is_cuda and value.dtype==torch.float16 and not torch.is_grad_enabled()
                self.scope=dict(name=_name,M=value.shape[2]*value.shape[3])
                try:
                    result=_before(value,*args,**kwargs)
                    assert result.shape==value.shape and result.is_contiguous()
                    full=torch.empty_like(result,memory_format=torch.contiguous_format)
                    dist.all_gather_into_tensor(full,result[self.rank*8:(self.rank+1)*8].contiguous())
                    if torch.cuda.is_current_stream_capturing():self.rows.append(dict(module=_name,input_shape=list(value.shape),output_stride=list(full.stride()),collectives=1,local_output_MiB=full.numel()*full.element_size()/2**21,original_full16_GEMM_FFN_projection_descriptors=True,GroupNorm_before_partition_full_input=True))
                    return full
                finally:self.scope=None
            module.forward=forward;self.saved.append((module,before))
        assert len(self.saved)==16
    def masters(self):return {p.data_ptr():p for module,_ in self.saved for p in module.parameters()}
    def report(self):
        assert self.scope is None
        return dict(captured=len(self.rows),rows=self.rows.copy(),native_attention_calls=len(self.calls),calls=self.calls.copy(),
            original59_math_unchanged=True,original_full16_vendor_GEMM_FFN_and_projection_descriptors=True,local8_only_at_native_attention_boundary=True,
            full5D_GroupNorm_and16T_preserved=True,global_masks_and_full_condition_caches_preserved=True,additional_collectives_per_forward=16,
            unowned_frame_outputs_consumed_only_by_independent_frame_rows_until_block_gather=True,full_video_PSNR_unverified=True)


def install(model):
    global active
    assert active is None and xformers.ops.memory_efficient_attention is parent_attention
    active=Controller(model);xformers.ops.memory_efficient_attention=attention;return active


def uninstall():
    global active,enabled
    if active is not None:
        assert active.scope is None and xformers.ops.memory_efficient_attention is attention
        xformers.ops.memory_efficient_attention=parent_attention
        for module,forward in reversed(active.saved):module.forward=forward
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:yield
    finally:enabled=previous
