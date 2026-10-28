"""Experimental two-product FP16 GEMM; FP32 masters, accumulation and output.

Only one residual cross product is retained. This deliberately changes numerical
precision and requires a new complete trajectory check before adoption.
"""
import torch
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import _half_split
from .gemm_output_allocation import empty as empty_gemm_output


@triton.jit
def _two_product_gemm(X, High, Low, Bias, Output,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      HAS_BIAS: tl.constexpr, INPUT_PROTECTED: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles = tl.cdiv(M, BM)
    col_tiles = tl.cdiv(N, BN)
    group = pid // (GROUP_M * col_tiles)
    first_row = group * GROUP_M
    group_rows = tl.minimum(GROUP_M, row_tiles - first_row)
    within_group = pid % (GROUP_M * col_tiles)
    rows = (first_row + within_group % group_rows) * BM + tl.arange(0, BM)
    cols = (within_group // group_rows) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        if INPUT_PROTECTED:
            x_high, x_low = _half_split(x, 1.0)
        else:
            x_high = x.to(tl.float16)
        w_high = tl.load(High + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
        primary = _rounding_boundary(tl.dot(x_high, w_high))
        if INPUT_PROTECTED:
            correction = tl.dot(x_low, w_high)
        else:
            w_low = tl.load(Low + cols[None, :] * K + k[:, None],
                            (cols[None, :] < N) & (k[:, None] < K), other=0)
            correction = tl.dot(x_high, w_low)
        block = _rounding_boundary(primary + correction * (1 / 2048.0))
        accumulator += block * (1 / 256.0)
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    tl.store(Output + rows[:, None] * N + cols[None, :], accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))


def two_product_linear(value, weight_parts, bias=None, *, protect,
                       debug_poison=False):
    if protect not in {'input', 'weight'}:
        raise ValueError('Protection must be input or weight')
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim < 1 or not value.is_contiguous() or len(weight_parts) != 2):
        raise ValueError('Two-product GEMM requires contiguous CUDA FP32 inference input')
    high, low = weight_parts
    if (high.ndim != 2 or value.shape[-1] != high.shape[1]
            or any(p.shape != high.shape or p.device != value.device
                   or p.dtype != torch.float16 or not p.is_contiguous()
                   for p in weight_parts)):
        raise ValueError('Invalid two-product packed weights')
    channels, reduction = high.shape
    if bias is not None and (bias.device != value.device or bias.dtype != torch.float32
                            or bias.shape != (channels,) or not bias.is_contiguous()):
        raise ValueError('Invalid two-product FP32 bias')
    rows = value.numel() // reduction
    if min(rows, channels, reduction) <= 0:
        raise ValueError('Empty two-product GEMM is unsupported')
    output = empty_gemm_output('classic', (*value.shape[:-1], channels),
                               device=value.device, dtype=torch.float32)
    if debug_poison:
        output.fill_(float('nan'))
    _two_product_gemm[(triton.cdiv(rows, 64) * triton.cdiv(channels, 64),)](
        value, high, low, high if bias is None else bias, output,
        rows, channels, reduction, bias is not None, protect == 'input',
        64, 64, 64, 32, num_warps=4, num_stages=2, enable_fp_fusion=False)
    return output
