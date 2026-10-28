"""Scoped single-part normalization dispatch; original modules and masters stay."""
from collections import Counter
from contextlib import contextmanager
import torch
from unifolm_wma.utils import triton_split_group_norm as original
from unifolm_wma.utils.triton_serial_group_norm import fused

ORIGINAL=original.split_group_norm_silu
_controller=None


def measure(candidate,baseline):
    assert candidate.shape==baseline.shape and candidate.dtype==baseline.dtype==torch.float32
    assert torch.isfinite(candidate).all() and torch.isfinite(baseline).all()
    bits=torch.equal(candidate.contiguous().view(torch.int32),baseline.contiguous().view(torch.int32))
    row=dict(signedzero_bits=bits,max_abs=(candidate-baseline).abs().max().item(),relative_l2=((candidate-baseline).norm()/baseline.norm().clamp_min(1e-30)).item())
    assert bits,row
    return row


class Controller:
    def __init__(self):self.enabled=False;self.captures={};self.fallbacks=Counter()

    def forward(self,value,norm,*,apply_silu=True,native_forward=None):
        reason=None
        if not self.enabled:reason='disabled'
        elif torch.is_grad_enabled() or norm.training or torch.is_autocast_enabled():reason='training_or_autocast'
        elif not value.is_cuda or value.dtype!=torch.float32 or value.ndim not in (4,5) or value.numel()==0:reason='input'
        elif value.shape[1]!=norm.num_channels or getattr(norm,'_inference_dtype',torch.float32)!=torch.float32:reason='norm'
        elif tuple(value.shape) not in ((16, 320, 40, 64), (16, 640, 20, 32), (16, 1280, 10, 16), (16, 1280, 5, 8)) or not value.is_contiguous() or norm.num_groups!=32:reason='unqualified_geometry'
        elif any(p is None or p.dtype!=value.dtype or p.device!=value.device or not p.is_contiguous() for p in (norm.weight,norm.bias)):reason='affine'
        if reason is not None:
            self.fallbacks[reason]+=1
            return ORIGINAL(value,norm,apply_silu=apply_silu,native_forward=native_forward)
        if torch.cuda.is_current_stream_capturing():
            key=(id(norm),value.data_ptr(),apply_silu)
            self.captures[key]=dict(shape=list(value.shape),strides=list(value.stride()),groups=norm.num_groups,size=value.numel()//(value.shape[0]*norm.num_groups),silu=apply_silu,input_ptr=value.data_ptr(),affine_ptrs=[norm.weight.data_ptr(),norm.bias.data_ptr()])
        return fused(value,norm,apply_silu=apply_silu)

    def report(self):
        return dict(captured=len(self.captures),rows=list(self.captures.values()),fallbacks=dict(self.fallbacks),original_modules_and_master_storage=True,original4096_reduction_layout_preserved=True,FP32_stats_affine_SiLU_preserved=True,serial_original_partials_vector1024_output=True,qualified_contiguous_geometries_only=True,original_unqualified_fallback=True)


def install(selection=None):
    global _controller
    assert _controller is None and original.split_group_norm_silu is ORIGINAL
    _controller=Controller();original.split_group_norm_silu=_controller.forward
    return _controller


@contextmanager
def mode(enabled):
    assert _controller is not None
    prior=_controller.enabled;_controller.enabled=enabled
    try:yield _controller
    finally:_controller.enabled=prior


def uninstall():
    global _controller
    original.split_group_norm_silu=ORIGINAL;_controller=None


import json
import os
FLAG='UNIFOLM_SERIAL_GROUP_NORM'


def configured(environment,mode,dataset):
    raw=environment.get(FLAG,'0')
    if raw not in ('0','1'):raise ValueError('Invalid single-part GroupNorm flag')
    if raw=='0' or dataset!='unitree_z1_dual_arm_stackbox_v2':return False
    if mode!='native':raise ValueError('Single-part GroupNorm requires native FP32')
    if environment.get('UNIFOLM_SPLIT_GROUP_NORM')!='fp32':raise ValueError('Single-part GroupNorm requires split FP32')
    return True


def configure_serial_norm(model,*,mode,dataset):
    if not configured(os.environ,mode,dataset):return None
    if hasattr(model,'_single_part_norm_controller'):raise RuntimeError('Single-part GroupNorm already installed')
    controller=install();controller.enabled=True
    model._single_part_norm_controller=controller
    print('SERIAL_GROUP_NORM_INSTALLED '+json.dumps(dict(enabled=True,FP32_live_affines=True,original_modules_and_masters=True,original4096_reduction_layout=True)),flush=True)
    return controller


def report_capture():
    if _controller is None:return
    row=_controller.report();row['stage']='decision' if torch.backends.cudnn.allow_tf32 else 'world'
    print('SERIAL_GROUP_NORM_CAPTURE '+json.dumps(row),flush=True)
