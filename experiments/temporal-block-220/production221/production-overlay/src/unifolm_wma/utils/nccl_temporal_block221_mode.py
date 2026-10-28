"""Expand existing world FFN row ownership across temporal block computation.

Complete five-dimensional GroupNorm and final proj_out stay replicated. The
original FFN all-gather remains the only collective. Each spatial position keeps
all sixteen time steps. Original JIT bodies, reduction order and tiles survive.
"""
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import torch
import torch.distributed as dist
import triton
from . import nccl_initial_ffn220_mode as adopted
from unifolm_wma.utils import nccl_spatial_decision209_mode as spatial
from unifolm_wma.utils.nccl_geglu_209_mode import measure

parent_raw=spatial.parent_run
enabled=False
active=None


def row_view(value,m,width,rank):
    assert value.is_contiguous() and value.numel()==m*width
    return value.view(m,width)[rank*(m//2):(rank+1)*(m//2)]


def grouped_route(values,rank,scratch):
    """Pure row routing; a shared scratch retains both predicate launch results."""
    result=dict(values);m=values['M'];n=values['N'];k=values['K'];part=m//2
    assert m%32==0 and values['OUTPUT_GROUPS']==3 and n%3==0
    assert tuple(scratch.shape)==(3,part,n//3) and scratch.is_contiguous()
    for name in ('XH','XL'):result[name]=row_view(values[name],m,k,rank)
    result['M']=part;result['Y']=scratch
    grid=(triton.cdiv(part,values['BM'])*triton.cdiv(n,values.get('BN',64)),)
    return result,grid


def dispatch(jit,args,kwargs,scope,rank,parent):
    values=dict(zip(jit.arg_names,args));values.update({k:v for k,v in kwargs.items() if k in jit.arg_names})
    name=jit.fn.__name__;m=scope['M'];channels=scope['C'];part=m//2
    # Existing FFN hooks already changed M and gathered their output. Never
    # halve those launches again or add a second collective.
    if 'M' in values and values['M']!=m:return parent(jit,*args,**kwargs)
    if name in {'_zero_low_gemm','_zero_low_gemm_persistent','_gemm32'} and values.get('OUTPUT_GROUPS')==3:
        assert values['K']==channels and values['N']==3*channels
        assert kwargs.get('enable_fp_fusion') is False
        output=values['Y'];assert output.is_contiguous() and output.numel()==3*m*channels
        key=(output.data_ptr(),m,channels)
        scratch=scope['scratch'].get(key)
        if scratch is None:
            scratch=torch.empty((3,part,channels),dtype=output.dtype,device=output.device);scope['scratch'][key]=scratch
        routed,grid=grouped_route(values,rank,scratch)
        if name=='_zero_low_gemm_persistent' and values['THREE']:grid=(min(grid[0],216),)
        options={k:v for k,v in kwargs.items() if k not in jit.arg_names and k!='grid'}
        result=parent(jit,*[routed[k] for k in jit.arg_names],grid=grid,**options)
        # One strided copy; include all scratch allocation/copy costs in timing.
        # Both predicate launches use the SAME buffer: an inactive second
        # launch must never overwrite the active first result with fresh memory.
        output.view(3,m,channels)[:,rank*part:(rank+1)*part].copy_(scratch)
        if torch.cuda.is_current_stream_capturing():scope['shaders'][name]+=1;scope['grouped_copy_nodes']+=1
        return result
    if name=='_attention_fp32' and jit.fn.__module__=='unifolm_wma.utils.triton_temporal_attention_fp32':
        assert not values['DEBUG'] and values['HEADS']*64==channels
        for name in ('Q','K','V','Output'):values[name]=row_view(values[name],m,channels,rank)
        options={k:v for k,v in kwargs.items() if k not in jit.arg_names and k!='grid'}
        result=parent(jit,*[values[k] for k in jit.arg_names],grid=(part//16*values['HEADS'],),**options)
        if torch.cuda.is_current_stream_capturing():scope['shaders']['_attention_fp32']+=1
        return result
    # All remaining supported row kernels use the already validated spatial
    # router. Spatial-only attention/layout kernels are excluded explicitly.
    allowed={'_presplit_two_product_gemm','_packed_attention_output_gemm','_half_compensated_gemm','_two_product_gemm',
        'inline_zero_low_gemm','_zero_low_gemm','_zero_low_gemm_persistent','_residual_layer_norm',
        '_residual_norm_half_pack','_pack_half_weight'}
    if name in allowed:return spatial.dispatch(jit,args,kwargs,scope,rank,parent)
    return parent(jit,*args,**kwargs)


@contextmanager
def new_shape_cache(rank):
    assert 'TRITON_CACHE_MANAGER' not in os.environ
    previous=os.environ.get('TRITON_CACHE_DIR');root=Path(os.environ['TEMPORAL_BLOCK_CACHE_DIR']);assert root.is_dir()
    directory=root/('rank-'+str(rank));directory.mkdir(mode=0o700,exist_ok=True)
    os.environ['TRITON_CACHE_DIR']=str(directory)
    try:yield
    finally:
        if previous is None:os.environ.pop('TRITON_CACHE_DIR',None)
        else:os.environ['TRITON_CACHE_DIR']=previous


def launch(jit,*args,**kwargs):
    if not enabled or active is None or active.scope is None or not active.scope['armed'] or kwargs.get('warmup',False):
        return parent_raw(jit,*args,**kwargs)
    assert not torch.is_grad_enabled() and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    with new_shape_cache(active.rank):return dispatch(jit,args,kwargs,active.scope,active.rank,parent_raw)


spatial.parent_run=launch


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import TemporalTransformer
        self.base=adopted.install(model);self.rank=dist.get_rank();self.scope=None;self.saved=[];self.rows=[]
        for name,module in model.named_modules():
            if type(module) is not TemporalTransformer:continue
            # init_attn gathers the raw FFN output BEFORE its residual add.
            # Its residual remains replicated until that gather is moved.
            if not module.use_linear:continue
            assert module.only_self_att and not module.causal_attention and len(module.transformer_blocks)==1
            before=module.forward
            def forward(value,*args,_before=before,_module=module,_name=name,**kwargs):
                if not enabled or torch.backends.cudnn.allow_tf32:return _before(value,*args,**kwargs)
                assert self.scope is None and value.ndim==5 and value.shape[0]==1 and value.shape[2]==16
                channels=_module.transformer_blocks[0].attn1.to_q.in_features
                self.scope=dict(owner=_module,name=_name,M=16*value.shape[3]*value.shape[4],C=channels,
                    armed=False,shaders=Counter(),scratch={},grouped_copy_nodes=0)
                try:
                    result=_before(value,*args,**kwargs)
                    assert not self.scope['armed'] and result.shape==value.shape
                    if torch.cuda.is_current_stream_capturing():
                        self.rows.append(dict(module=_name,M=self.scope['M'],C=channels,shaders=dict(self.scope['shaders']),
                            grouped_copy_nodes=self.scope['grouped_copy_nodes'],additional_collectives=0,
                            full5D_GroupNorm_preserved=True,all16_time_steps_preserved=True,proj_out_after_existing_FFN_gather=True))
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
                if enabled and self.scope is not None and self.scope['owner'] is _module:self.scope['armed']=False
                return result
            block.forward=finish;self.saved.append((block,before))
        assert len(self.saved)==16*3
    @property
    def captures(self):return self.base.captures
    def report(self):
        assert self.scope is None
        report=self.base.report()
        report['temporal_block']=dict(captured=len(self.rows),rows=self.rows.copy(),
            no_new_collectives=True,original_JIT_bodies_K_order_and_tiles=True,
            original_full_dispatch_shapes=True,full5D_GroupNorm_and_complete16_time_steps_preserved=True,
            grouped_scratch_and_copy_costs_included=True,init_attn_uses_existing220_only=True)
        return report


def install(model):
    global active
    assert active is None and spatial.parent_run is launch;active=Controller(model);return active


def uninstall():
    global active,enabled
    assert spatial.parent_run is launch;spatial.parent_run=parent_raw
    if active is not None:
        assert active.scope is None
        for module,forward in reversed(active.saved):module.forward=forward
        adopted.uninstall()
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:
        with adopted.mode(True):yield
    finally:assert active.scope is None;enabled=previous
