"""Fuse exact-GELU activation and gated multiplication for FP16 inference."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _geglu(Projected, Output, N: tl.constexpr, CHANNELS: tl.constexpr,
            BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    source = (offset // CHANNELS) * (2 * CHANNELS) + offset % CHANNELS
    value = tl.load(Projected + source, valid, other=0).to(tl.float32)
    gate = tl.load(Projected + source + CHANNELS, valid, other=0).to(tl.float32)
    gelu = (gate * 0.5) * (1. + tl.math.erf(gate * 0.7071067811865476))
    # Eager F.gelu materializes a half tensor before the gated multiplication.
    gelu = gelu.to(tl.float16).to(tl.float32)
    tl.store(Output + offset, value * gelu, valid)


def fused_geglu(projected):
    if (not projected.is_cuda or projected.dtype != torch.float16 or
            not projected.is_contiguous() or torch.is_grad_enabled()):
        value, gate = projected.chunk(2, dim=-1)
        return value * F.gelu(gate)
    channels = projected.shape[-1] // 2
    if projected.shape[-1] != 2 * channels:
        raise ValueError("GEGLU requires an even projected dimension")
    output = torch.empty((*projected.shape[:-1], channels),
                         device=projected.device, dtype=projected.dtype)
    count = output.numel()
    if count:
        _geglu[(triton.cdiv(count, 1024),)](
            projected, output, count, channels, 1024,
            num_warps=4, enable_fp_fusion=False)
    return output
