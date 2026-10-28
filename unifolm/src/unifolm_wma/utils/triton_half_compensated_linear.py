"""Experimental three-product FP16 GEMM with scaled residuals and FP32 output.

This is approximate arithmetic with a narrower exponent range than TF32.
Inference installation requires an explicit opt-in. Actual-input and full-video checks
are required before any adoption; scales must keep both split parts finite.
"""
import torch
from .gemm_output_allocation import empty as empty_gemm_output
from .half_pack_allocation import empty_like as empty_half_pack
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary


@triton.jit
def _half_split(value, SCALE: tl.constexpr):
    scaled = value * SCALE
    high = scaled.to(tl.float16)
    low = ((scaled - high.to(tl.float32)) * 2048.0).to(tl.float16)
    return high, low


@triton.jit
def _half_compensated_gemm(X, Weight, WeightLow, Bias, Output,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                           BN: tl.constexpr, XS: tl.constexpr, WS: tl.constexpr,
                           PACKED: tl.constexpr, GROUP_M: tl.constexpr = 0,
                           BK: tl.constexpr = 32):
    if GROUP_M:
        pid = tl.program_id(0)
        row_tiles = tl.cdiv(M, BM)
        col_tiles = tl.cdiv(N, BN)
        group = pid // (GROUP_M * col_tiles)
        first_row = group * GROUP_M
        group_rows = tl.minimum(GROUP_M, row_tiles - first_row)
        within_group = pid % (GROUP_M * col_tiles)
        row_tile = first_row + within_group % group_rows
        col_tile = within_group // group_rows
    else:
        row_tile = tl.program_id(0)
        col_tile = tl.program_id(1)
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        x_high, x_low = _half_split(x, XS)
        weight = tl.load(Weight + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
        if PACKED:
            w_high = weight
            w_low = tl.load(WeightLow + cols[None, :] * K + k[:, None],
                            (cols[None, :] < N) & (k[:, None] < K), other=0)
        else:
            w_high, w_low = _half_split(weight, WS)
        primary = _rounding_boundary(tl.dot(x_high, w_high))
        correction = tl.dot(x_low, w_high)
        correction = tl.dot(x_high, w_low, correction)
        block = _rounding_boundary(primary + correction * (1 / 2048.0))
        accumulator += block * (1 / (XS * WS))
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    tl.store(Output + rows[:, None] * N + cols[None, :], accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _pack_half_weight(Weight, High, Low, SIZE: tl.constexpr,
                      SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Weight + offset, offset < SIZE, other=0)
    high, low = _half_split(value, SCALE)
    tl.store(High + offset, high, offset < SIZE)
    tl.store(Low + offset, low, offset < SIZE)


def pack_half_weight(weight, *, weight_scale=256.0, out=None):
    if (not weight.is_cuda or weight.dtype != torch.float32 or
            not weight.is_contiguous() or weight.ndim != 2 or torch.is_grad_enabled()):
        raise ValueError('Packing requires contiguous CUDA FP32 inference weights')
    parts = out if out is not None else tuple(
        empty_half_pack(weight) for _ in range(2))
    if len(parts) != 2 or any(p.shape != weight.shape or p.device != weight.device or
                              p.dtype != torch.float16 or not p.is_contiguous()
                              for p in parts):
        raise ValueError('Invalid packed weight storage')
    _pack_half_weight[(triton.cdiv(weight.numel(), 1024),)](
        weight, *parts, weight.numel(), weight_scale, 1024,
        num_warps=4, enable_fp_fusion=False)
    import os
    correction_mode=os.environ.get('UNIFOLM_DECISION_VENDOR_CORRECTION','0')
    if correction_mode not in {'0','1'}:
        raise ValueError('Invalid decision vendor correction mode')
    if correction_mode=='1' and weight_scale==256. and tuple(weight.shape)==(640,2560):
        from .vendor_correction_count import refresh_count
        refresh_count(parts)
    return parts


def half_compensated_linear(value, weight, bias=None, *, block_m=32, block_n=64,
                            num_warps=4, x_scale=1.0, weight_scale=256.0,
                            weight_parts=None, group_m=0, block_k=32):
    if group_m not in {0, 4, 8, 16, 32}:
        raise ValueError('Unsupported GEMM row tile group')
    if block_k not in {32, 64, 128}:
        raise ValueError('Unsupported compensated GEMM reduction tile')
    if (not value.is_cuda or value.dtype != torch.float32 or
            weight.dtype != torch.float32 or value.device != weight.device or
            not value.is_contiguous() or not weight.is_contiguous() or
            value.shape[-1] != weight.shape[1] or torch.is_grad_enabled()):
        raise ValueError('Diagnostic half-compensated GEMM requires CUDA FP32 inference')
    if bias is not None and (bias.device != value.device or bias.dtype != torch.float32
                            or bias.shape != (weight.shape[0],) or not bias.is_contiguous()):
        raise ValueError('Invalid diagnostic half-compensated bias')
    rows = value.numel() // value.shape[-1]
    channels, reduction = weight.shape
    if weight_parts is not None and (len(weight_parts) != 2 or any(
            p.shape != weight.shape or p.device != weight.device or
            p.dtype != torch.float16 or not p.is_contiguous() for p in weight_parts)):
        raise ValueError('Invalid packed compensated weight')
    high, low = (weight, weight) if weight_parts is None else weight_parts
    result = empty_gemm_output('classic',(*value.shape[:-1], channels), device=value.device, dtype=value.dtype)
    row_tiles, col_tiles = triton.cdiv(rows, block_m), triton.cdiv(channels, block_n)
    grid = (row_tiles * col_tiles,) if group_m else (row_tiles, col_tiles)
    _half_compensated_gemm[grid](
        value, high, low, weight if bias is None else bias, result,
        rows, channels, reduction, bias is not None, block_m, block_n,
        x_scale, weight_scale, weight_parts is not None, GROUP_M=group_m, BK=block_k,
        num_warps=num_warps, num_stages=2,
        enable_fp_fusion=False)
    return result


from .decision_zero_low_runtime import wrap_pack as _decision_zero_low_wrap_pack
pack_half_weight=_decision_zero_low_wrap_pack(pack_half_weight)

from .decision_zero_low_runtime import wrap_inline as _decision_zero_low_wrap_inline
_half_compensated_gemm=_decision_zero_low_wrap_inline(_half_compensated_gemm)


from .world_zero_low_runtime import wrap_pack as _world_zero_low_wrap_pack
pack_half_weight=_world_zero_low_wrap_pack(pack_half_weight)
