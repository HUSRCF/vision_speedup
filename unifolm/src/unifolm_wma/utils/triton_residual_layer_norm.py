"""Inference residual addition and LayerNorm in a single CUDA launch."""
import torch
from . import norm_buffer_allocation as _norm_allocation
import triton
import triton.language as tl


@triton.jit
def _residual_layer_norm(Left, Right, Weight, Bias, Residual, Normalized,
                         CHANNELS: tl.constexpr, EPS: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < CHANNELS
    offset = row * CHANNELS + channel
    left = tl.load(Left + offset, valid, other=0).to(tl.float32)
    right = tl.load(Right + offset, valid, other=0).to(tl.float32)
    # Preserve the rounding boundary of the separately materialized sum.
    residual = (left + right).to(Left.dtype.element_ty)
    value = residual.to(tl.float32)
    mean = tl.sum(value, 0) / CHANNELS
    centered = tl.where(valid, value - mean, 0.)
    variance = tl.sum(centered * centered, 0) / CHANNELS
    inverse = tl.math.rsqrt(variance + EPS)
    weight = tl.load(Weight + channel, valid, other=0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0).to(tl.float32)
    normalized = ((value - mean) * inverse) * weight + bias
    tl.store(Residual + offset, residual, valid)
    tl.store(Normalized + offset, normalized, valid)


def fused_residual_layer_norm(left, right, norm):
    if (not left.is_cuda or left.shape != right.shape or
            left.device != right.device or left.dtype != right.dtype or
            left.dtype not in (torch.float16, torch.float32) or
            not left.is_contiguous() or not right.is_contiguous() or
            left.ndim == 0 or tuple(norm.normalized_shape) != (left.shape[-1],) or
            left.shape[-1] == 0 or left.shape[-1] > 8192 or
            norm.weight is None or norm.bias is None or
            norm.weight.device != left.device or norm.bias.device != left.device or
            norm.weight.dtype != left.dtype or norm.bias.dtype != left.dtype or
            not norm.weight.is_contiguous() or not norm.bias.is_contiguous() or
            norm.training or torch.is_grad_enabled()):
        residual = left + right
        return residual, norm(residual)
    channels = left.shape[-1]
    residual = _norm_allocation.empty_like('residual', left)
    normalized = _norm_allocation.empty_like('residual', left)
    rows = left.numel() // channels
    if rows:
        _residual_layer_norm[(rows,)](
            left, right, norm.weight, norm.bias, residual, normalized,
            channels, norm.eps, triton.next_power_of_2(channels),
            num_warps=4, enable_fp_fusion=False)
    return residual, normalized
