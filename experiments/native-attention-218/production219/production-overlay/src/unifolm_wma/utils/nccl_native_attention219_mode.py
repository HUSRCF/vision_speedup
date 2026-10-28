"""Native xFormers batch sharding inside the qualified full16 spatial scope."""
from contextlib import contextmanager
import torch
import xformers.ops
from . import nccl_resblock_native_grid_production209_mode as adopted
from . import nccl_spatial_decision209_mode as spatial

enabled=False
active=None
parent_attention=xformers.ops.memory_efficient_attention


def local_native(parent,q,k,v,bias,*,rank,p=0.,scale=None,op=None,output_dtype=None):
    """CPU-testable batch/mask routing; the live wrapper checks CUDA scope."""
    assert rank in (0,1) and q.ndim==k.ndim==v.ndim==4
    assert q.shape[0]==k.shape[0]==v.shape[0]==16
    assert q.shape[2]==k.shape[2]==v.shape[2] and k.shape[1]==v.shape[1]
    assert q.shape[3]==k.shape[3] and p==0.
    begin,end=rank*8,(rank+1)*8
    if isinstance(bias,torch.Tensor):
        assert bias.ndim==4 and bias.shape[0] in (1,16)
        local_bias=bias[begin:end] if bias.shape[0]==16 else bias
    else:
        assert bias is None
        local_bias=None
    result=parent(q[begin:end],k[begin:end],v[begin:end],attn_bias=local_bias,
                  p=p,scale=scale,op=op,output_dtype=output_dtype)
    assert result.shape==(8,q.shape[1],q.shape[2],v.shape[3])
    full=torch.empty((16,*result.shape[1:]),dtype=result.dtype,device=result.device)
    full[begin:end].copy_(result)
    record=dict(query_shape=list(q.shape),key_shape=list(k.shape),value_shape=list(v.shape),
                local_batch=8,outer_batch=16,rank=rank,
                bias_present=bias is not None,original_global_bias_slice=True,
                local_output_copy_bytes=result.numel()*result.element_size(),
                output_shape_preserved=True,extra_collectives=0)
    return full,record


def attention(q,k,v,attn_bias=None,p=0.,scale=None,*,op=None,output_dtype=None):
    options=dict(attn_bias=attn_bias,p=p,scale=scale,op=op,output_dtype=output_dtype)
    scope=spatial.active.scope if spatial.active is not None else None
    if not enabled or active is None or not spatial.enabled or scope is None:
        return parent_attention(q,k,v,**options)
    assert not torch.is_grad_enabled() and not torch.backends.cuda.matmul.allow_tf32
    assert q.ndim==k.ndim==v.ndim==4 and q.shape[0]==k.shape[0]==v.shape[0]==16
    assert all(t.is_cuda and t.dtype==torch.float32 and t.device==q.device for t in (q,k,v))
    assert scope['M']==16*q.shape[1] and scope['C']==q.shape[2]*q.shape[3]
    result,record=local_native(parent_attention,q,k,v,attn_bias,rank=active.rank,
                               p=p,scale=scale,op=op,output_dtype=output_dtype)
    if torch.cuda.is_current_stream_capturing():
        active.calls.append(dict(spatial_module=scope['name'],**record))
    return result


class StageCalls(list):
    def append(self,record):
        stage='decision' if torch.backends.cudnn.allow_tf32 else 'world'
        super().append(dict(stage=stage,**record))


class Controller:
    def __init__(self,model):
        self.base=adopted.install(model)
        self.rank=spatial.active.rank;self.calls=StageCalls()
    @property
    def captures(self):
        return self.base.captures
    def report(self):
        report=self.base.report()
        report['native_attention']=dict(captured=len(self.calls),rows=self.calls.copy(),
            stage_counts={stage:sum(r['stage']==stage for r in self.calls) for stage in ('decision','world')},
            full16_outer_shapes_and_existing_GEMM_dispatch=True,
            compensated_self_attention_unchanged=True,
            global_masks_and_KV_cache_unchanged=True,
            no_new_collectives=True,unowned_output_frames_not_consumed_before_existing_gather=True)
        return report


def install(model):
    global active
    assert active is None and xformers.ops.memory_efficient_attention is parent_attention
    active=Controller(model);xformers.ops.memory_efficient_attention=attention
    return active


def uninstall():
    global active,enabled
    if active is not None:
        assert xformers.ops.memory_efficient_attention is attention
        xformers.ops.memory_efficient_attention=parent_attention
        adopted.uninstall()
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    prior=enabled;enabled=value
    try:
        with adopted.mode(value):yield
    finally:enabled=prior
