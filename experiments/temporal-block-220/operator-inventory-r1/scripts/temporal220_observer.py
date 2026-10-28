"""Observe existing temporal JIT metadata on adopted220 without GPU arg edits."""
from contextlib import contextmanager
import os
from pathlib import Path
import torch
import unifolm_wma.utils as package
utils=Path(os.environ['ADOPTED220_ROOT'])/'src/unifolm_wma/utils'
assert utils.is_dir();package.__path__.append(str(utils))
from unifolm_wma.utils import nccl_initial_ffn220_mode as adopted
from unifolm_wma.utils import nccl_spatial_decision209_mode as spatial
from unifolm_wma.utils.nccl_geglu_209_mode import measure

parent_raw=spatial.parent_run
enabled=False
active=None
SCALARS={'M','N','K','C','T','H','W','B','BATCH','HEADS','HEAD_DIM','BM','BN','BK','GROUP','GROUP_M','CHANNELS','SIZE','BLOCK','OUTPUT_GROUPS','TEMPORAL','S0','S1','ST','SH','SW'}


def describe(jit,args,kwargs):
    values=dict(zip(jit.arg_names,args));values.update({k:v for k,v in kwargs.items() if k in jit.arg_names})
    tensors={};scalars={}
    for name,value in values.items():
        if isinstance(value,torch.Tensor):
            tensors[name]=dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),contiguous=value.is_contiguous())
        elif name in SCALARS and type(value) in (bool,int,float):scalars[name]=value
    grid=kwargs.get('grid')
    if isinstance(grid,(tuple,list)) and all(type(v) is int for v in grid):grid=list(grid)
    else:grid='callable_or_unspecified'
    return dict(kernel=jit.fn.__name__,module=jit.fn.__module__,tensors=tensors,scalars=scalars,grid=grid,
        num_warps=kwargs.get('num_warps'),num_stages=kwargs.get('num_stages'),enable_fp_fusion=kwargs.get('enable_fp_fusion'))


def observe(jit,*args,**kwargs):
    if enabled and active is not None and active.scope is not None and not kwargs.get('warmup',False) and torch.cuda.is_current_stream_capturing():
        row=describe(jit,args,kwargs);row['phase']=active.scope['phase'];active.scope['JIT_calls'].append(row)
    # Always pass the same argument objects, launch grid and options to original.
    return parent_raw(jit,*args,**kwargs)


spatial.parent_run=observe


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import TemporalTransformer
        self.base=adopted.install(model);self.scope=None;self.saved=[];self.rows=[]
        for name,module in model.named_modules():
            if type(module) is not TemporalTransformer:continue
            assert module.only_self_att and not module.causal_attention
            before=module.forward
            def forward(value,*args,_before=before,_module=module,_name=name,**kwargs):
                if not enabled:return _before(value,*args,**kwargs)
                assert self.scope is None and value.ndim==5 and value.shape[2]==16
                self.scope=dict(owner=_module,module_name=_name,input_shape=list(value.shape),input_stride=list(value.stride()),
                    use_linear=_module.use_linear,inner_channels=_module.transformer_blocks[0].attn1.to_q.in_features,phase='norm',JIT_calls=[])
                try:
                    result=_before(value,*args,**kwargs)
                    assert result.shape==value.shape and result.dtype==value.dtype
                    self.scope.update(output_shape=list(result.shape),output_stride=list(result.stride()))
                    if torch.cuda.is_current_stream_capturing():self.rows.append({k:v for k,v in self.scope.items() if k!='owner'})
                    return result
                finally:self.scope=None
            module.forward=forward;self.saved.append((module,before))
            for phase,layer in (('norm',module.norm),('proj_in',module.proj_in),('proj_out',module.proj_out)):
                before=layer.forward
                def boundary(value,*args,_before=before,_module=module,_phase=phase,**kwargs):
                    scope=self.scope
                    if not enabled or scope is None or scope['owner'] is not _module:return _before(value,*args,**kwargs)
                    previous=scope['phase'];scope['phase']=_phase
                    try:return _before(value,*args,**kwargs)
                    finally:scope['phase']='body' if _phase=='norm' else previous
                layer.forward=boundary;self.saved.append((layer,before))
        assert len(self.saved)==17*4
    @property
    def captures(self):return self.base.captures
    def report(self):
        assert self.scope is None
        return dict(captured=len(self.rows),rows=self.rows.copy(),adopted220=self.base.report(),
            original_GPU_argument_objects_grids_options_unchanged=True,
            no_tensor_values_addresses_or_driver_parameters_exported=True,
            only_shape_stride_dtype_and_selected_dimensions=True,
            no_kernel_or_precision_change=True,no_performance_or_full_video_qualification=True)


def install(model):
    global active
    assert active is None and spatial.parent_run is observe
    active=Controller(model);return active


def uninstall():
    global active,enabled
    assert spatial.parent_run is observe;spatial.parent_run=parent_raw
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
