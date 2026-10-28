"""Diagnostic one-product linear: inline input conversion and FP32 epilogue."""
import torch
from .gemm_output_allocation import empty as empty_gemm_output
import triton
import triton.language as tl


@triton.jit
def _primary(X, WH, Bias, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
             GROUP: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (GROUP * col_tiles)
    first = group * GROUP
    count = tl.minimum(GROUP, row_tiles - first)
    local = pid % (GROUP * col_tiles)
    rows = (first + local % count) * BM + tl.arange(0, BM)
    cols = (local // count) * BN + tl.arange(0, BN)
    inner = tl.arange(0, 64)
    acc = tl.full((BM, BN), 0., tl.float32)
    for block in range(tl.cdiv(K, 64)):
        k = block * 64 + inner
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0).to(tl.float16)
        wh = tl.load(WH + cols[None, :] * K + k[:, None],
                     (cols[None, :] < N) & (k[:, None] < K), other=0)
        acc = tl.dot(x, wh, acc)
    acc *= 1. / 256.
    if HAS_BIAS:
        acc += tl.load(Bias + cols, cols < N, other=0)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def primary_narrow_linear(value, high_weight, bias=None, *, tile=(64, 64, 32, 4),
                          debug_poison=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim < 1 or not value.is_contiguous()
            or high_weight.ndim != 2 or high_weight.dtype != torch.float16
            or high_weight.device != value.device or not high_weight.is_contiguous()
            or value.shape[-1] != high_weight.shape[1]):
        raise ValueError('Expected CUDA contiguous FP32 input and scaled high Half weights')
    rows = value.numel() // value.shape[-1]
    outputs, inner = high_weight.shape
    if min(rows, outputs, inner) <= 0:
        raise ValueError('Empty shapes unsupported')
    if bias is not None and (bias.shape != (outputs,) or bias.dtype != torch.float32
            or bias.device != value.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    bm, bn, group, warps = tile
    if (bm not in {32, 64, 128} or bn not in {64, 128} or group not in {16, 32}
            or warps not in {4, 8}):
        raise ValueError('Unsupported primary narrow tile')
    result = empty_gemm_output('primary',(*value.shape[:-1], outputs), device=value.device, dtype=torch.float32)
    if debug_poison:
        result.fill_(float('nan'))
    _primary[(triton.cdiv(rows, bm) * triton.cdiv(outputs, bn),)](
        value, high_weight, high_weight if bias is None else bias, result,
        rows, outputs, inner, bias is not None, bm, bn, group,
        num_warps=warps, num_stages=2, enable_fp_fusion=False)
    return result
