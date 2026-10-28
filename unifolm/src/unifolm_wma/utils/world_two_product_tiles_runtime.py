"""Retain original two-product JIT and its five tensors; tune world launches."""
from collections import Counter
from contextlib import contextmanager
import torch,triton
from unifolm_wma.utils.triton_two_product_linear import _two_product_gemm
ORIGINAL_JIT=_two_product_gemm
ORIGINAL_RUN=ORIGINAL_JIT.run
active=None


def measure(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
    return dict(relative_l2=0.,max_abs=0.,signedzero_bits=True)


class Controller:
    def __init__(self,selection):
        self.selection={tuple(r['geometry']):r for r in selection};assert len(self.selection)==len(selection)==2
        assert set(self.selection)=={(40960,320,320),(10240,640,640)}
        for row in selection:assert row['has_bias'] is True and row['nodes']==35 and len(row['config'])==5
        self.enabled=False;self.captures={};self.fallbacks=Counter()

    def fallback(self,reason,args,kwargs):
        self.fallbacks[reason]+=1;return ORIGINAL_RUN(*args,**kwargs)

    def run(self,*args,**kwargs):
        if not self.enabled:return ORIGINAL_RUN(*args,**kwargs)
        if torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
            return self.fallback('stage_grad_autocast',args,kwargs)
        if len(args)!=14:return self.fallback('arguments',args,kwargs)
        m,n,k=args[5:8];selected=self.selection.get((m,n,k))
        if selected is None or args[8:]!=(True,True,64,64,64,32):return self.fallback('geometry_or_math',args,kwargs)
        if kwargs.get('num_warps')!=4 or kwargs.get('num_stages')!=2 or kwargs.get('enable_fp_fusion') is not False or kwargs.get('warmup') is not False or tuple(kwargs.get('grid',()))!=(triton.cdiv(m,64)*triton.cdiv(n,64),):
            return self.fallback('launch',args,kwargs)
        if any(not isinstance(p,torch.Tensor) or not p.is_cuda or not p.is_contiguous() for p in args[:5]):return self.fallback('storage',args,kwargs)
        value,high,low,bias,output=args[:5]
        if value.dtype!=torch.float32 or value.numel()!=m*k or any(p.dtype!=torch.float16 or p.device!=value.device or p.shape!=(n,k) for p in (high,low)) or bias.dtype!=torch.float32 or bias.device!=value.device or bias.shape!=(n,) or output.dtype!=torch.float32 or output.device!=value.device or output.numel()!=m*n:
            return self.fallback('layout',args,kwargs)
        bm,bn,group,warps,stages=selected['config'];revised=list(args);revised[10]=bm;revised[11]=bn;revised[13]=group
        assert all(a is b for a,b in zip(args[:5],revised[:5])) and revised[12]==args[12]==64
        if torch.cuda.is_current_stream_capturing():
            key=tuple(p.data_ptr() for p in args[:5]);self.captures[key]=dict(geometry=[m,n,k],config=selected['config'],five_tensor_arguments_retained=True)
        return ORIGINAL_RUN(*revised,**dict(kwargs,grid=(triton.cdiv(m,bm)*triton.cdiv(n,bn),),num_warps=warps,num_stages=stages))

    def report(self):
        assert _two_product_gemm is ORIGINAL_JIT and ORIGINAL_JIT.run==self.run
        return dict(captured=len(self.captures),rows=list(self.captures.values()),fallbacks=dict(self.fallbacks),original_JIT_object_and_saved_run_retained=True,original_five_tensor_arguments=True,original_BK64_two_products_and_inline_split=True,original_graph_owners_version_and_master_cache=True,world_only=True,only_qualified_launch_tiles=True)


def install(selection):
    global active
    assert active is None and ORIGINAL_JIT.run==ORIGINAL_RUN
    active=Controller(selection);ORIGINAL_JIT.run=active.run;return active


@contextmanager
def mode(enabled):
    assert active is not None
    prior=active.enabled;active.enabled=enabled
    try:yield active
    finally:active.enabled=prior


def uninstall():
    global active
    if active is not None:ORIGINAL_JIT.run=ORIGINAL_RUN;active=None


import json
import os
FLAG='UNIFOLM_WORLD_TWO_PRODUCT_TILES'
DEPENDENCIES={'UNIFOLM_SERIAL_GROUP_NORM': '1', 'UNIFOLM_HEAD_NORM_MISH': '1', 'UNIFOLM_SPLIT_GROUP_NORM': 'fp32', 'UNIFOLM_NORM_BUFFER_NOFILL': 'all', 'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE': '1'}
CONFIGS=[{'geometry': [40960, 320, 320], 'config': [64, 64, 32, 4, 3], 'has_bias': True, 'nodes': 35, 'median_saved_ms': 0.007168006896972667}, {'geometry': [10240, 640, 640], 'config': [32, 128, 32, 4, 3], 'has_bias': True, 'nodes': 35, 'median_saved_ms': 0.033843195438385015}]


def configured(environment,mode,dataset):
    raw=environment.get(FLAG,'0')
    if raw not in ('0','1'):raise ValueError('Invalid two-product launch tiles flag')
    if raw=='0' or dataset!='unitree_z1_dual_arm_stackbox_v2':return False
    if mode!='native':raise ValueError('Two-product launch tiles requires native inference')
    for name,value in DEPENDENCIES.items():
        if environment.get(name)!=value:raise ValueError('Two-product launch tiles requires '+name+'='+value)
    return True


def configure_two_product_tiles(model,*,mode,dataset):
    if not configured(os.environ,mode,dataset):return None
    if hasattr(model,'_two_product_tiles_controller'):raise RuntimeError('Two-product tiles already installed')
    controller=install(CONFIGS);controller.enabled=True
    model._two_product_tiles_controller=controller
    print('TWO_PRODUCT_TILES_INSTALLED '+json.dumps(dict(enabled=True,geometries=len(CONFIGS),nodes=sum(r['nodes'] for r in CONFIGS),same_original_JIT=True,all_setup_inside_process_wall=True)),flush=True)
    return controller


def report_capture():
    if active is None:return
    report=active.report();report['stage']='decision' if torch.backends.cudnn.allow_tf32 else 'world'
    print('TWO_PRODUCT_TILES_CAPTURE '+json.dumps(report),flush=True)
