"""Explicit per-call packed residual norm output; no dynamic result cache."""
import torch
import triton
from .triton_residual_norm_half_pack_kernel import _residual_norm_half_pack
from . import norm_buffer_allocation,half_pack_allocation


def residual_norm_half_pack(left,right,norm,*,debug=False):
    if (torch.is_grad_enabled() or torch.is_autocast_enabled() or not left.is_cuda
            or left.dtype!=torch.float32 or right.dtype!=left.dtype
            or left.shape!=right.shape or left.device!=right.device
            or left.ndim==0 or not left.is_contiguous() or not right.is_contiguous()
            or left.shape[-1]<=0 or left.shape[-1]>8192 or left.numel()==0
            or tuple(norm.normalized_shape)!=(left.shape[-1],) or norm.training
            or norm.weight is None or norm.bias is None
            or any(p.dtype!=torch.float32 or p.device!=left.device or not p.is_contiguous()
                   for p in [norm.weight,norm.bias])):
        raise ValueError('Expected contiguous CUDA FP32 residual LayerNorm inference')
    channels=left.shape[-1];rows=left.numel()//channels
    residual=norm_buffer_allocation.empty_like('residual',left)
    normalized=norm_buffer_allocation.empty_like('residual',left) if debug else residual
    inputs=tuple(half_pack_allocation.empty_like(left.view(rows,channels)) for _ in range(2))
    _residual_norm_half_pack[(rows,)](
        left,right,norm.weight,norm.bias,residual,normalized,*inputs,
        channels,norm.eps,triton.next_power_of_2(channels),debug,
        num_warps=4,enable_fp_fusion=False)
    return (residual,*inputs,normalized) if debug else (residual,*inputs)
