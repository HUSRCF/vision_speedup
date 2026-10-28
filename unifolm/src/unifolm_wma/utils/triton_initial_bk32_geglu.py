"""Fuse initial FFN projection with its existing BK32 and native GELU math."""
import torch
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary
from .triton_native_geglu_activation import native_geglu
from .geglu_output_allocation import empty as empty_geglu


@triton.jit
def _piece32(xh, xl, wh, wl):
    primary = _rounding_boundary(tl.dot(xh, wh))
    correction = tl.dot(xl, wh)
    correction = tl.dot(xh, wl, correction)
    return _rounding_boundary(primary + correction * (1. / 2048.)) * (1. / 256.)


@triton.jit
def _bk32_native_geglu(X, WH, WL, Bias, Y, V, G,
                       M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                       HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr, GROUP: tl.constexpr, DEBUG: tl.constexpr):
    pid = tl.program_id(0)
    mt, nt = tl.cdiv(M, BM), tl.cdiv(C, BN)
    group = pid // (GROUP * nt)
    first = group * GROUP
    size = tl.minimum(GROUP, mt - first)
    local = pid % (GROUP * nt)
    rows = (first + local % size) * BM + tl.arange(0, BM)
    cols = (local // size) * BN + tl.arange(0, BN)
    inner = tl.arange(0, 32)
    value = tl.full((BM, BN), 0., tl.float32)
    gate = tl.full((BM, BN), 0., tl.float32)
    for block in range(tl.cdiv(K, 32)):
        k = block * 32 + inner
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        xh, xl = _half_split(x, 1.)
        offset = cols[None, :] * K + k[:, None]
        valid = (cols[None, :] < C) & (k[:, None] < K)
        vh = tl.load(WH + offset, valid, other=0)
        vl = tl.load(WL + offset, valid, other=0)
        gh = tl.load(WH + offset + C * K, valid, other=0)
        gl = tl.load(WL + offset + C * K, valid, other=0)
        value += _piece32(xh, xl, vh, vl)
        gate += _piece32(xh, xl, gh, gl)
    if HAS_BIAS:
        value += tl.load(Bias + cols, cols < C, other=0)[None, :]
        gate += tl.load(Bias + cols + C, cols < C, other=0)[None, :]
    output = rows[:, None] * C + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < C)
    if DEBUG:
        tl.store(V + output, value, valid)
        tl.store(G + output, gate, valid)
    tl.store(Y + output, native_geglu(value, gate), valid)


def initial_bk32_geglu(value, parts, bias=None, *, tile=(64, 32, 32, 4, 2),
                        debug_branches=False, debug_poison=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim < 1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous CUDA FP32 inference input')
    if len(parts) != 2 or any(p.ndim != 2 or p.shape != parts[0].shape
            or p.shape[1] != value.shape[-1] or p.shape[0] % 2
            or p.device != value.device or p.dtype != torch.float16
            or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid cached GEGLU weight parts')
    rows, inner = value.numel() // value.shape[-1], value.shape[-1]
    channels = parts[0].shape[0] // 2
    if min(rows, inner, channels) <= 0:
        raise ValueError('Empty GEGLU dimensions unsupported')
    if bias is not None and (bias.shape != (2 * channels,) or bias.dtype != torch.float32
            or bias.device != value.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 GEGLU bias')
    bm, bn, group, warps, stages = tile
    if (bm not in {32, 64} or bn not in {32, 64} or group != 32
            or warps not in {4, 8} or stages not in {1, 2}):
        raise ValueError('Unsupported initial BK32 GEGLU tile')
    shape = (*value.shape[:-1], channels)
    output = empty_geglu(shape, device=value.device, dtype=torch.float32)
    v = torch.empty_like(output) if debug_branches else output
    g = torch.empty_like(output) if debug_branches else output
    if debug_poison:
        for tensor in (output, v, g): tensor.fill_(float('nan'))
    _bk32_native_geglu[(triton.cdiv(rows, bm) * triton.cdiv(channels, bn),)](
        value, *parts, parts[0] if bias is None else bias, output, v, g,
        rows, channels, inner, bias is not None, bm, bn, group, debug_branches,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return (output, v, g) if debug_branches else output
