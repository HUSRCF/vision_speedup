"""Move existing initial FFN gather after residual; shard prior init attention.

Full5D norm and Conv1d boundary projections remain replicated. No additional
collective or changed arithmetic; candidate extends the same row ownership.
"""
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import torch
import torch.distributed as dist
from . import nccl_temporal_block221_mode as adopted
from . import nccl_spatial_decision209_mode as spatial
from . import nccl_initial_ffn220_mode as initial
parent_run=adopted.parent_raw
enabled=False
active=None


class DeferredInitialGather:
    def __init__(self,rank,gather):
        assert rank in (0,1);self.rank=rank;self.gather=gather;self.pending=None;self.completed=0
    def record(self,output,local):
        assert self.pending is None and output.shape==(40960,512) and output.is_contiguous()
        rows=slice(self.rank*20480,(self.rank+1)*20480)
        assert local.shape==(20480,512) and local.is_contiguous()
        assert local.dtype==output.dtype==torch.float32 and local.device==output.device
        assert local.data_ptr()==output[rows].data_ptr()
        self.pending=(output,local)
    def finish(self,result):
        assert self.pending is not None and result.is_contiguous() and result.numel()==40960*512
        previous,local=self.pending;assert result.dtype==previous.dtype and result.device==previous.device
        output=result.view(40960,512);rows=slice(self.rank*20480,(self.rank+1)*20480)
        # The caller supplies the completed block result including residual.
        # Retaining the old raw output until here prevents allocator reuse.
        assert output.data_ptr()!=previous.data_ptr()
        self.gather(output,output[rows]);self.pending=None;self.completed+=1
        return result


def launch(jit,*args,**kwargs):
    if not enabled or active is None or active.scope is None or not active.scope['armed'] or kwargs.get('warmup',False):return parent_run(jit,*args,**kwargs)
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_grad_enabled()
    with adopted.new_shape_cache(active.rank):
        return adopted.dispatch(jit,args,kwargs,active.scope,active.rank,parent_run)


adopted.parent_raw=launch


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import TemporalTransformer
        self.base=adopted.install(model);self.rank=dist.get_rank();self.scope=None;self.saved=[];self.rows=[]
        self.initial=initial.active.initial;self.original_gather=self.initial.gather
        self.deferred=DeferredInitialGather(self.rank,self.original_gather)
        def gather(output,local):
            if not enabled or self.scope is None or not self.scope['armed']:return self.original_gather(output,local)
            return self.deferred.record(output,local)
        self.initial.gather=gather
        for name,module in model.named_modules():
            if type(module) is not TemporalTransformer or module.use_linear:continue
            assert module.only_self_att and not module.causal_attention and len(module.transformer_blocks)==1
            assert module.transformer_blocks[0].attn1.to_q.in_features==512
            before=module.forward
            def forward(value,*args,_before=before,_module=module,_name=name,**kwargs):
                if not enabled or torch.backends.cudnn.allow_tf32:return _before(value,*args,**kwargs)
                assert self.scope is None and value.shape==(1,320,16,40,64) and self.deferred.pending is None
                self.scope=dict(owner=_module,name=_name,M=40960,C=512,armed=False,shaders=Counter(),scratch={},grouped_copy_nodes=0)
                try:
                    result=_before(value,*args,**kwargs)
                    assert not self.scope['armed'] and self.deferred.pending is None
                    if torch.cuda.is_current_stream_capturing():self.rows.append(dict(module=_name,stage='world',shaders=dict(self.scope['shaders']),grouped_copy_nodes=self.scope['grouped_copy_nodes'],existing_gather_moved_after_residual=True,additional_collectives=0))
                    return result
                finally:self.scope=None
            module.forward=forward;self.saved.append((module,before))
            before=module.norm.forward
            def norm(value,*args,_before=before,_module=module,**kwargs):
                result=_before(value,*args,**kwargs)
                if enabled and self.scope is not None and self.scope['owner'] is _module:self.scope['armed']=True
                return result
            module.norm.forward=norm;self.saved.append((module.norm,before))
            block=module.transformer_blocks[-1];before=block.forward
            def finish(*args,_before=before,_module=module,**kwargs):
                result=_before(*args,**kwargs)
                if enabled and self.scope is not None and self.scope['owner'] is _module:
                    result=self.deferred.finish(result);self.scope['armed']=False
                return result
            block.forward=finish;self.saved.append((block,before))
        assert len(self.saved)==3
    @property
    def captures(self):return self.base.captures
    def report(self):
        assert self.scope is None and self.deferred.pending is None
        report=self.base.report()
        report['initial_block']=dict(captured=len(self.rows),rows=self.rows.copy(),
            no_new_collectives=True,initial_collective_contains_residual=True,
            original_JIT_bodies_K_order_and_tiles=True,full5D_norm_and16_time_steps_preserved=True,
            Conv1d_boundary_projections_stay_replicated=True)
        report['temporal_block']['init_attn_uses_existing220_only']=False
        return report



def install(model):
    global active
    assert active is None and adopted.parent_raw is launch;active=Controller(model);return active


def uninstall():
    global active,enabled
    assert adopted.parent_raw is launch;adopted.parent_raw=parent_run
    if active is not None:
        assert active.scope is None and active.deferred.pending is None
        active.initial.gather=active.original_gather
        for module,forward in reversed(active.saved):module.forward=forward
        adopted.uninstall()
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:
        with adopted.mode(True):yield
    finally:
        assert active.scope is None and active.deferred.pending is None
        enabled=previous
