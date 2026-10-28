"""Diagnostic direct NCHW patch splitting with deterministic split-K GEMM."""
import torch; from .triton_zero_weight_low_spatial import run_packed_components, run_f2_components, residual_counts
from .conv_buffer_allocation import empty as empty_conv_buffer
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split, pack_half_weight
from .triton_compensated_linear import _rounding_boundary


@triton.jit
def _pack(X, High, Low, M: tl.constexpr, C: tl.constexpr,
          H: tl.constexpr, W: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)[:, None]
    k = tl.program_id(1) * BC + tl.arange(0, BC)[None, :]
    batch = rows // (H * W)
    y, x = (rows // W) % H + (k % 9) // 3 - 1, rows % W + k % 3 - 1
    source = ((batch * C + k // 9) * H + y) * W + x
    valid = (rows < M) & (k < C * 9)
    value = tl.load(X + source, valid & (y >= 0) & (y < H) & (x >= 0) & (x < W), other=0)
    high, low = _half_split(value, 1.)
    offset = rows * (C * 9) + k
    tl.store(High + offset, high, valid)
    tl.store(Low + offset, low, valid)


@triton.jit
def _gemm(XH, XL, WH, WL, Bias, Output, M: tl.constexpr, N: tl.constexpr,
          K: tl.constexpr, S: tl.constexpr, HAS_BIAS: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          SPLITS: tl.constexpr, GROUP: tl.constexpr):
    pid, split = tl.program_id(0), tl.program_id(1)
    row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (GROUP * col_tiles)
    first = group * GROUP
    count = tl.minimum(GROUP, row_tiles - first)
    local = pid % (GROUP * col_tiles)
    rows = (first + local % count) * BM + tl.arange(0, BM)
    cols = (local // count) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0., tl.float32)
    steps = tl.cdiv(tl.cdiv(K, BK), SPLITS)
    for block in range(steps):
        k = (split * steps + block) * BK + inner
        mask_x = (rows[:, None] < M) & (k[None, :] < K)
        xh = tl.load(XH + rows[:, None] * K + k[None, :], mask_x, other=0)
        xl = tl.load(XL + rows[:, None] * K + k[None, :], mask_x, other=0)
        mask_w = (cols[None, :] < N) & (k[:, None] < K)
        wh = tl.load(WH + cols[None, :] * K + k[:, None], mask_w, other=0)
        wl = tl.load(WL + cols[None, :] * K + k[:, None], mask_w, other=0)
        primary = _rounding_boundary(tl.dot(xh, wh))
        correction = tl.dot(xl, wh)
        correction = tl.dot(xh, wl, correction)
        acc += _rounding_boundary(primary + correction * (1. / 2048.)) * (1. / 256.)
    if SPLITS == 1:
        if HAS_BIAS:
            acc += tl.load(Bias + cols, cols < N, other=0)[None, :]
        offset = ((rows[:, None] // S) * N + cols[None, :]) * S + rows[:, None] % S
    else:
        offset = split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(Output + offset, acc, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _finish(Partial, Bias, Output, M: tl.constexpr, N: tl.constexpr,
            S: tl.constexpr, SPLITS: tl.constexpr, HAS_BIAS: tl.constexpr,
            BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, channel = (i // (N * S)) * S + i % S, (i // S) % N
    acc = tl.full((BLOCK,), 0., tl.float32)
    for split in range(SPLITS):
        acc += tl.load(Partial + split * M * N + row * N + channel, i < M * N, other=0)
    if HAS_BIAS:
        acc += tl.load(Bias + channel, i < M * N, other=0)
    tl.store(Output + i, acc, i < M * N)


def packed_spatial_conv(value, parts, bias=None, *, block_m=64, block_n=64, splits=1):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim != 4 or not value.is_contiguous() or min(value.shape) <= 0):
        raise ValueError('Expected nonempty contiguous inference CUDA FP32 NCHW')
    batch, channels, height, width = value.shape
    if len(parts) != 2 or any(p.ndim != 2 or p.shape != parts[0].shape
            or p.shape[1] != channels * 9 or p.dtype != torch.float16
            or p.device != value.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid compensated spatial weight parts')
    outputs = parts[0].shape[0]
    if bias is not None and (bias.shape != (outputs,) or bias.dtype != torch.float32
            or bias.device != value.device or not bias.is_contiguous()):
        raise ValueError('Invalid spatial FP32 bias')
    if block_m not in {32, 64} or block_n not in {64, 128} or splits not in {1, 2, 4, 8}:
        raise ValueError('Unsupported spatial split-K configuration')
    rows, inner = batch * height * width, channels * 9
    packed = tuple(empty_conv_buffer('spatial',(rows, inner), device=value.device, dtype=torch.float16)
                   for _ in range(2))
    _pack[(triton.cdiv(rows, 32), triton.cdiv(inner, 32))](value, *packed,
        rows, channels, height, width, 32, 32, num_warps=4, enable_fp_fusion=False)
    output = empty_conv_buffer('spatial',(batch, outputs, height, width), device=value.device, dtype=torch.float32)
    partial = output if splits == 1 else empty_conv_buffer('spatial',(splits, rows, outputs),
        device=value.device, dtype=torch.float32)
    if not run_packed_components(packed,parts,bias,partial,rows,outputs,inner,height*width,block_m,block_n,splits):
        _gemm[(triton.cdiv(rows, block_m) * triton.cdiv(outputs, block_n), splits)](
            *packed, *parts, parts[0] if bias is None else bias, partial,
            rows, outputs, inner, height * width, bias is not None,
            block_m, block_n, 64, splits, 16, num_warps=4, num_stages=2,
            enable_fp_fusion=False)
    if splits != 1:
        _finish[(triton.cdiv(rows * outputs, 256),)](partial,
            parts[0] if bias is None else bias, output, rows, outputs,
            height * width, splits, bias is not None, 256, num_warps=4,
            enable_fp_fusion=False)
    return output


def spatial_weight_parts(weight, *, out=None):
    if weight.ndim != 4 or weight.shape[2:] != (3, 3):
        raise ValueError('Expected OI33 weights')
    parts=pack_half_weight(weight.flatten(1),out=out)
    residual_counts(parts,'packed',refresh=True)
    return parts
