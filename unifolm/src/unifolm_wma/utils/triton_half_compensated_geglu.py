"""Opt-in scaled FP16 three-product GEMM fused with FP32 GEGLU output."""
import torch
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary


@triton.jit
def _product(x_high, x_low, weight):
    w_high, w_low = _half_split(weight, 256.0)
    primary = _rounding_boundary(tl.dot(x_high, w_high))
    correction = tl.dot(x_low, w_high)
    correction = tl.dot(x_high, w_low, correction)
    return _rounding_boundary(primary + correction * (1 / 2048.0)) * (1 / 256.0)


@triton.jit
def _half_compensated_geglu(X, Weight, Bias, Output,
                            M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                            HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                            BN: tl.constexpr, GROUP_M: tl.constexpr):
    if GROUP_M:
        pid = tl.program_id(0)
        row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(C, BN)
        group = pid // (GROUP_M * col_tiles)
        first_row = group * GROUP_M
        group_rows = tl.minimum(GROUP_M, row_tiles - first_row)
        local = pid % (GROUP_M * col_tiles)
        row_tile = first_row + local % group_rows
        col_tile = local // group_rows
    else:
        row_tile, col_tile = tl.program_id(0), tl.program_id(1)
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, 32)
    value_acc = tl.full((BM, BN), 0, tl.float32)
    gate_acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, 32)):
        k = start * 32 + inner
        value = tl.load(X + rows[:, None] * K + k[None, :],
                        (rows[:, None] < M) & (k[None, :] < K), other=0)
        vw = tl.load(Weight + cols[None, :] * K + k[:, None],
                     (cols[None, :] < C) & (k[:, None] < K), other=0)
        gw = tl.load(Weight + (cols[None, :] + C) * K + k[:, None],
                     (cols[None, :] < C) & (k[:, None] < K), other=0)
        x_high, x_low = _half_split(value, 1.0)
        value_acc += _product(x_high, x_low, vw)
        gate_acc += _product(x_high, x_low, gw)
    if HAS_BIAS:
        value_acc += tl.load(Bias + cols, cols < C, other=0)[None, :]
        gate_acc += tl.load(Bias + C + cols, cols < C, other=0)[None, :]
    gelu = (gate_acc * 0.5) * (1. + tl.math.erf(gate_acc * 0.7071067811865476))
    tl.store(Output + rows[:, None] * C + cols[None, :], value_acc * gelu,
             (rows[:, None] < M) & (cols[None, :] < C))


def half_compensated_geglu(value, weight, bias=None, *, block_m=64, block_n=32, group_m=0):
    if group_m not in {0, 4, 8, 16, 32}:
        raise ValueError('Unsupported Half GEGLU row grouping')
    if (not value.is_cuda or value.dtype != torch.float32 or
            weight.dtype != torch.float32 or value.device != weight.device or
            not value.is_contiguous() or not weight.is_contiguous() or
            value.shape[-1] != weight.shape[1] or weight.shape[0] % 2 or
            torch.is_grad_enabled()):
        raise ValueError('Half compensated GEGLU requires CUDA FP32 inference')
    if bias is not None and (bias.device != value.device or bias.dtype != torch.float32
                            or bias.shape != (weight.shape[0],) or not bias.is_contiguous()):
        raise ValueError('Invalid compensated GEGLU bias')
    rows = value.numel() // value.shape[-1]
    channels, reduction = weight.shape[0] // 2, weight.shape[1]
    output = torch.empty((*value.shape[:-1], channels), device=value.device,
                         dtype=torch.float32)
    row_tiles, col_tiles = triton.cdiv(rows, block_m), triton.cdiv(channels, block_n)
    grid = (row_tiles*col_tiles,) if group_m else (row_tiles, col_tiles)
    _half_compensated_geglu[grid](
        value, weight, weight if bias is None else bias, output,
        rows, channels, reduction, bias is not None, block_m, block_n, group_m,
        num_warps=4, num_stages=2, enable_fp_fusion=False)
    return output
