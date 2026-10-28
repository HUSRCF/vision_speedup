"""Diagnostic C320 GEGLU with individually protected BK32 value/gate math."""
import torch
from .geglu_output_allocation import empty as empty_geglu_output
import triton
import triton.language as tl
from .triton_half_compensated_linear import pack_half_weight
from .triton_prepacked_geglu import _short_product


@triton.jit
def _geglu(XH, XL, WH, WL, Bias, Y, DebugValue, DebugGate,
           M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
           HAS_BIAS: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr,
           PRIMARY_VALUE: tl.constexpr, PRIMARY_GATE: tl.constexpr,
           DEBUG: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles, col_tiles = tl.cdiv(M, 64), tl.cdiv(C, BN)
    group = pid // (GROUP * col_tiles)
    first = group * GROUP
    count = tl.minimum(GROUP, row_tiles - first)
    local = pid % (GROUP * col_tiles)
    rows = (first + local % count) * 64 + tl.arange(0, 64)
    cols = (local // count) * BN + tl.arange(0, BN)
    inner = tl.arange(0, 32)
    value = tl.full((64, BN), 0., tl.float32)
    gate = tl.full((64, BN), 0., tl.float32)
    for block in range(tl.cdiv(K, 32)):
        k = block * 32 + inner
        xoffset = rows[:, None] * K + k[None, :]
        xvalid = (rows[:, None] < M) & (k[None, :] < K)
        xh = tl.load(XH + xoffset, xvalid, other=0)
        if not (PRIMARY_VALUE and PRIMARY_GATE):
            xl = tl.load(XL + xoffset, xvalid, other=0)
        woffset = cols[None, :] * K + k[:, None]
        wvalid = (cols[None, :] < C) & (k[:, None] < K)
        vh = tl.load(WH + woffset, wvalid, other=0)
        gh = tl.load(WH + woffset + C * K, wvalid, other=0)
        if PRIMARY_VALUE:
            value = tl.dot(xh, vh, value)
        else:
            vl = tl.load(WL + woffset, wvalid, other=0)
            value += _short_product(xh, xl, vh, vl)
        if PRIMARY_GATE:
            gate = tl.dot(xh, gh, gate)
        else:
            gl = tl.load(WL + woffset + C * K, wvalid, other=0)
            gate += _short_product(xh, xl, gh, gl)
    if PRIMARY_VALUE:
        value *= 1. / 256.
    if PRIMARY_GATE:
        gate *= 1. / 256.
    if HAS_BIAS:
        value += tl.load(Bias + cols, cols < C, other=0)[None, :]
        gate += tl.load(Bias + cols + C, cols < C, other=0)[None, :]
    offset = rows[:, None] * C + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < C)
    if DEBUG:
        tl.store(DebugValue + offset, value, valid)
        tl.store(DebugGate + offset, gate, valid)
    gelu = (gate * .5) * (1. + tl.math.erf(gate * .7071067811865476))
    tl.store(Y + offset, value * gelu, valid)


def primary_narrow_geglu(value, parts, bias=None, *, mode='value', block_n=32,
                         debug_branches=False, debug_poison=False):
    if mode not in {'compensated', 'value', 'gate', 'both'} or block_n not in {32, 64}:
        raise ValueError('Unsupported narrow GEGLU experiment')
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim < 1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous inference CUDA FP32 input')
    if len(parts) != 2 or any(p.ndim != 2 or p.shape != parts[0].shape
            or p.shape[1] != value.shape[-1] or p.shape[0] % 2
            or p.dtype != torch.float16 or p.device != value.device
            or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid GEGLU half weight parts')
    rows = value.numel() // value.shape[-1]
    channels, inner = parts[0].shape[0] // 2, value.shape[-1]
    if min(rows, channels, inner) <= 0:
        raise ValueError('Empty shapes unsupported')
    if bias is not None and (bias.shape != (channels * 2,) or bias.dtype != torch.float32
            or bias.device != value.device or not bias.is_contiguous()):
        raise ValueError('Invalid GEGLU FP32 bias')
    inputs = pack_half_weight(value.view(rows, inner), weight_scale=1.)
    output = empty_geglu_output((*value.shape[:-1], channels), device=value.device, dtype=torch.float32)
    v = torch.empty_like(output) if debug_branches else output
    g = torch.empty_like(output) if debug_branches else output
    if debug_poison:
        output.fill_(float('nan'))
    _geglu[(triton.cdiv(rows, 64) * triton.cdiv(channels, block_n),)](
        *inputs, *parts, parts[0] if bias is None else bias, output, v, g,
        rows, channels, inner, bias is not None, block_n, 16,
        mode in {'value', 'both'}, mode in {'gate', 'both'}, debug_branches,
        num_warps=4, num_stages=2, enable_fp_fusion=False)
    return (output, v, g) if debug_branches else output
