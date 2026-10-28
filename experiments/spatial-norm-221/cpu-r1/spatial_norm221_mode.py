"""Pending spatial-only norm row routing on frozen temporal221.

No temporal GroupNorm is partitioned. All original JIT arguments other than
row pointers and launch grids stay unchanged, including full M and strides.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import torch
import unifolm_wma.utils as package
root=Path(os.environ['ADOPTED221_ROOT'])/'src/unifolm_wma/utils';assert root.is_dir();package.__path__.append(str(root))
from unifolm_wma.utils import nccl_temporal_block221_mode as adopted
from unifolm_wma.utils import nccl_spatial_decision209_mode as spatial
from unifolm_wma.utils.nccl_geglu_209_mode import measure
parent_dispatch=spatial.dispatch
enabled=False
active=None


def route_norm(name,values,grid,scope,rank):
    """Pure routing of independent N16 samples; preserve every scalar exactly."""
    assert rank in (0,1);v=dict(values);m=scope['M'];c=scope['C'];rows=slice(rank*8,(rank+1)*8)
    assert m%16==0;area=m//16;groups=32
    def sample(slot):
        x=v[slot];assert x.ndim==4 and x.shape[0]==16 and x.shape[1]==c and x.numel()==m*c
        v[slot]=x[rows]
    def stats(slot,parts=False):
        x=v[slot];assert x.is_contiguous() and x.shape[0]==16*groups
        if parts:assert x.shape==(16*groups,v['PARTS'],2)
        else:assert x.shape==(16*groups,2)
        v[slot]=x[rank*8*groups:(rank+1)*8*groups]
    def output(slot):
        x=v[slot];assert x.is_contiguous() and x.numel()==m*c
        v[slot]=x.view(16,-1)[rows]
    if name in ('_partial','_apply','_serial_norm'):
        assert v['FRAMES']==1 and v['CHANNELS']==c and v['GROUPS']==groups and v['HEIGHT']*v['WIDTH']==area
        sample('X');assert grid[0]==16*groups
        if name=='_partial':stats('Stats',True)
        elif name=='_apply':stats('Moments');output('Output')
        else:
            assert name=='_serial_norm' and not v['DEBUG'];output('Output')
        grid=(8*groups,*grid[1:])
    elif name=='_finish_stats':
        assert v['SIZE']==c//groups*area and grid==(16*groups,)
        stats('Stats',True);stats('Moments');grid=(8*groups,)
    else:
        assert name=='_apply_layout_pack' and v['TEMPORAL'] is False
        assert (v['M'],v['C'],v['G'],v['T'],v['H']*v['W'])==(m,c,groups,1,area)
        assert (m//2)%v['RM']==0 and grid[0]==m//v['RM']
        sample('X');stats('Moments')
        for slot in ('PH','PL'):output(slot)
        if v['DEBUG']:output('Debug')
        grid=(grid[0]//2,*grid[1:])
    return v,grid


def dispatch(jit,args,kwargs,scope,rank,parent):
    if not enabled or active is None or spatial.active is None or scope is not spatial.active.scope:
        return parent_dispatch(jit,args,kwargs,scope,rank,parent)
    name=jit.fn.__name__;module=jit.fn.__module__
    allowed=(module=='unifolm_wma.utils.triton_split_group_norm' and name in ('_partial','_finish_stats','_apply')
        or module=='unifolm_wma.utils.triton_serial_group_norm' and name=='_serial_norm'
        or module=='unifolm_wma.utils.triton_norm_projection_pack' and name=='_apply_layout_pack')
    if not allowed:return parent_dispatch(jit,args,kwargs,scope,rank,parent)
    values=dict(zip(jit.arg_names,args));values.update({k:v for k,v in kwargs.items() if k in jit.arg_names})
    assert not kwargs.get('warmup',False) and kwargs.get('enable_fp_fusion') is False
    routed,grid=route_norm(name,values,kwargs['grid'],scope,rank)
    options={k:v for k,v in kwargs.items() if k not in jit.arg_names and k!='grid'}
    result=parent(jit,*[routed[k] for k in jit.arg_names],grid=grid,**options)
    if torch.cuda.is_current_stream_capturing():
        active.rows.append(dict(module=scope['name'],kernel=name,stage='decision' if torch.backends.cudnn.allow_tf32 else 'world',
            original_scalar_dimensions_and_strides=True,original_JIT_body=True,grid_row_axis_halved=True))
    return result


spatial.dispatch=dispatch


class Controller:
    def __init__(self,model):self.base=adopted.install(model);self.rows=[]
    @property
    def captures(self):return self.base.captures
    def report(self):
        return dict(captured=len(self.rows),rows=self.rows.copy(),adopted221=self.base.report(),
            no_new_collectives=True,no_temporal_GroupNorm_partition=True,
            full_shape_owners_and_all_statistics_math_preserved=True,full_video_PSNR_unverified=True)


def install(model):
    global active
    assert active is None and spatial.dispatch is dispatch;active=Controller(model);return active


def uninstall():
    global active,enabled
    assert spatial.dispatch is dispatch;spatial.dispatch=parent_dispatch
    if active is not None:adopted.uninstall()
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:
        with adopted.mode(True):yield
    finally:enabled=previous
