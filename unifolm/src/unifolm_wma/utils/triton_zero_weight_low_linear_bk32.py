"""Prepacked initial512 GEMM retaining every accepted BK32 rounding boundary."""
import torch
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import pack_half_weight
from .gemm_output_allocation import empty as empty_gemm_output


@triton.jit
def _sum32(XH, XL, WH, WL, rows, cols,
           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, THREE: tl.constexpr):
    inner = tl.arange(0, 32)
    total = tl.full((BM, 64), 0., tl.float32)
    for block in range(tl.cdiv(K, 32)):
        k = block * 32 + inner
        xo = rows[:, None] * K + k[None, :]
        valid_x = (rows[:, None] < M) & (k[None, :] < K)
        xh = tl.load(XH + xo, valid_x, other=0)
        xl = tl.load(XL + xo, valid_x, other=0)
        wo = cols[None, :] * K + k[:, None]
        valid_w = (cols[None, :] < N) & (k[:, None] < K)
        wh = tl.load(WH + wo, valid_w, other=0)
        primary = _rounding_boundary(tl.dot(xh, wh))
        correction = tl.dot(xl, wh)
        if THREE:
            wl = tl.load(WL + wo, valid_w, other=0)
            correction = tl.dot(xh, wl, correction)
        total += _rounding_boundary(primary + correction * (1. / 2048.)) * (1. / 256.)
    return total


@triton.jit
def _gemm32(XH, XL, WH, WL, Count, Bias, Y,
            M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            HAS_BIAS: tl.constexpr, BM: tl.constexpr, GROUP: tl.constexpr,
            OUTPUT_GROUPS: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(N, 64)
    group = pid // (GROUP * col_tiles)
    first = group * GROUP
    size = tl.minimum(GROUP, row_tiles - first)
    local = pid % (GROUP * col_tiles)
    rows = (first + local % size) * BM + tl.arange(0, BM)
    cols = (local // size) * 64 + tl.arange(0, 64)
    if tl.load(Count) != 0:
        total = _sum32(XH, XL, WH, WL, rows, cols, M, N, K, BM, True)
    else:
        total = _sum32(XH, XL, WH, WL, rows, cols, M, N, K, BM, False)
    if HAS_BIAS:
        total += tl.load(Bias + cols, cols < N, other=0)[None, :]
    if OUTPUT_GROUPS == 1:
        offset = rows[:, None] * N + cols[None, :]
    else:
        width = N // OUTPUT_GROUPS
        offset = ((cols[None, :] // width) * M * width +
                  rows[:, None] * width + cols[None, :] % width)
    tl.store(Y + offset, total,
             (rows[:, None] < M) & (cols[None, :] < N))


def zero_weight_low_linear_bk32(value, parts, count, bias=None, *, block_m=64,
                               group_m=32, num_stages=2, debug_poison=False,
                               output_groups=1):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim < 1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous CUDA FP32 inference input')
    if len(parts) != 2 or any(p.ndim != 2 or p.shape != parts[0].shape
            or p.shape[1] != value.shape[-1] or p.device != value.device
            or p.dtype != torch.float16 or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid static half weight parts')
    if count.shape != () or count.dtype != torch.int64 or count.device != value.device:
        raise ValueError('Invalid graph-visible weight predicate')
    rows = value.numel() // value.shape[-1]
    channels, reduction = parts[0].shape
    if min(rows, channels, reduction) <= 0:
        raise ValueError('Empty GEMM unsupported')
    if output_groups not in {1, 3} or channels % output_groups:
        raise ValueError('Output groups must be 1 or 3 and divide channels')
    if bias is not None and (bias.shape != (channels,) or bias.dtype != torch.float32
            or bias.device != value.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    if block_m not in {32, 64} or group_m not in {16, 32} or num_stages not in {2, 3}:
        raise ValueError('Unsupported initial512 GEMM tile')
    inputs = pack_half_weight(value.view(rows, reduction), weight_scale=1.)
    shape = ((*value.shape[:-1], channels) if output_groups == 1 else
             (output_groups, *value.shape[:-1], channels // output_groups))
    output = empty_gemm_output('classic', shape, device=value.device, dtype=torch.float32)
    if debug_poison:
        output.fill_(float('nan'))
    _gemm32[(triton.cdiv(rows, block_m) * triton.cdiv(channels, 64),)](
        *inputs, *parts, count, parts[0] if bias is None else bias, output,
        rows, channels, reduction, bias is not None, block_m, group_m, output_groups,
        num_warps=4, num_stages=num_stages, enable_fp_fusion=False)
    return output
