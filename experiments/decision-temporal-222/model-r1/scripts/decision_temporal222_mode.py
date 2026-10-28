"""Shard decision temporal row kernels while preserving vendor GEMM descriptors."""
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import torch
import torch.distributed as dist
from unifolm_wma.utils import nccl_initial_block222_mode as adopted
from unifolm_wma.utils import nccl_temporal_block221_mode as temporal
from unifolm_wma.utils import nccl_spatial_decision209_mode as spatial
from unifolm_wma.utils.nccl_geglu_209_mode import measure

active=None
enabled=False
parent=adopted.parent_run


@contextmanager
def cache(rank):
    assert 'TRITON_CACHE_MANAGER'not in os.environ
    root=Path(os.environ['DECISION_TEMPORAL_CACHE_DIR']);assert root.is_dir()
    p=root/('rank-'+str(rank));p.mkdir(mode=0o700,exist_ok=True)
    previous=os.environ.get('TRITON_CACHE_DIR');os.environ['TRITON_CACHE_DIR']=str(p)
    try:yield
    finally:
        if previous is None:os.environ.pop('TRITON_CACHE_DIR',None)
        else:os.environ['TRITON_CACHE_DIR']=previous


def launch(jit,*args,**kwargs):
    if not enabled or active is None or active.scope is None or not active.scope['armed']or kwargs.get('warmup',False):return parent(jit,*args,**kwargs)
    assert torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_grad_enabled()
    name=jit.fn.__name__;scope=active.scope
    with cache(active.rank):
        # The world router also handles grouped QKV and complete16-step FP32
        # attention. Decision-only primary FFN shaders are row independent.
        if name in {'_primary','_geglu','_primary_geglu_hidden_parts','_primary_geglu_hidden_high','_primary_qkv_pack'}:
            return spatial.dispatch(jit,args,kwargs,scope,active.rank,parent)
        return temporal.dispatch(jit,args,kwargs,scope,active.rank,parent)


adopted.parent_run=launch


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import TemporalTransformer
        self.base=adopted.install(model);self.rank=dist.get_rank();self.scope=None;self.saved=[];self.rows=[]
        for name,module in model.named_modules():
            if type(module)is not TemporalTransformer:continue
            assert module.only_self_att and not module.causal_attention and len(module.transformer_blocks)==1
            before=module.forward
            def forward(value,*args,_before=before,_module=module,_name=name,**kwargs):
                if not enabled or not torch.backends.cudnn.allow_tf32:return _before(value,*args,**kwargs)
                assert self.scope is None and value.ndim==5 and value.shape[0]==1 and value.shape[2]==16
                self.scope=dict(owner=_module,name=_name,M=16*value.shape[3]*value.shape[4],C=_module.transformer_blocks[0].attn1.to_q.in_features,armed=False,shaders=Counter(),scratch={},grouped_copy_nodes=0)
                try:
                    result=_before(value,*args,**kwargs);assert not self.scope['armed']and result.shape==value.shape
                    if torch.cuda.is_current_stream_capturing():self.rows.append(dict(module=_name,M=self.scope['M'],C=self.scope['C'],shaders=dict(self.scope['shaders']),grouped_copy_nodes=self.scope['grouped_copy_nodes'],collectives=1,full5D_GroupNorm_preserved=True,complete16_time_steps=True,vendor_GEMM_descriptors_unchanged=True))
                    return result
                finally:self.scope=None
            module.forward=forward;self.saved.append((module,before))
            before=module.norm.forward
            def norm(value,*args,_before=before,_module=module,**kwargs):
                result=_before(value,*args,**kwargs)
                if enabled and self.scope is not None and self.scope['owner']is _module:self.scope['armed']=True
                return result
            module.norm.forward=norm;self.saved.append((module.norm,before))
            block=module.transformer_blocks[-1];before=block.forward
            def finish(*args,_before=before,_module=module,**kwargs):
                result=_before(*args,**kwargs)
                if enabled and self.scope is not None and self.scope['owner']is _module:
                    assert self.scope['armed']and result.is_contiguous()and result.numel()==self.scope['M']*self.scope['C']
                    m,c=self.scope['M'],self.scope['C'];part=m//2;assert part%16==0
                    full=result.view(m,c);output=torch.empty_like(full)
                    dist.all_gather_into_tensor(output,full[self.rank*part:(self.rank+1)*part])
                    result=output.view_as(result);self.scope['armed']=False
                return result
            block.forward=finish;self.saved.append((block,before))
        assert len(self.saved)==51

    def report(self):
        assert self.scope is None
        return dict(captured=len(self.rows),rows=self.rows.copy(),adopted222=self.base.report(),full5D_norm_and16_time_steps_preserved=True,vendor_GEMM_descriptors_unchanged=True,additional_collectives_per_decision_forward=17)


def install(model):
    global active
    assert active is None and adopted.parent_run is launch;active=Controller(model);return active


def uninstall():
    global active,enabled
    if active is not None:
        assert active.scope is None
        for module,forward in reversed(active.saved):module.forward=forward
        adopted.uninstall()
    assert adopted.parent_run is launch;adopted.parent_run=parent;active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:
        with adopted.mode(True):yield
    finally:assert active.scope is None;enabled=previous
