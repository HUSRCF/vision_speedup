"""Device-predicated temporal convolution with unchanged short-BK64 sums."""
import torch
import triton
import triton.language as tl
from .conv_buffer_allocation import empty as empty_conv_buffer
from .triton_compensated_temporal_conv import _pack_temporal
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary
from .triton_zero_weight_low_linear import _short_sum


@triton.jit
def _implicit_sum(X, WH, WL, rows, cols, M: tl.constexpr, N: tl.constexpr,
                  C: tl.constexpr, T: tl.constexpr, S: tl.constexpr,
                  THREE: tl.constexpr):
    inner = tl.arange(0, 64)
    batch = rows // (T * S)
    frame, spatial = (rows // S) % T, rows % S
    acc = tl.full((64, 64), 0., tl.float32)
    for start in range(tl.cdiv(C * 3, 64)):
        k = start * 64 + inner
        channel, tap = k // 3, k % 3
        source_frame = frame[:, None] + tap[None, :] - 1
        source = ((batch[:, None] * C + channel[None, :]) * T + source_frame) * S + spatial[:, None]
        valid = ((rows[:, None] < M) & (k[None, :] < C * 3)
                 & (source_frame >= 0) & (source_frame < T))
        value = tl.load(X + source, valid, other=0)
        xh, xl = _half_split(value, 1.)
        offset = cols[None, :] * (C * 3) + k[:, None]
        valid_w = (cols[None, :] < N) & (k[:, None] < C * 3)
        wh = tl.load(WH + offset, valid_w, other=0)
        primary = _rounding_boundary(tl.dot(xh, wh))
        correction = tl.dot(xl, wh)
        if THREE:
            wl = tl.load(WL + offset, valid_w, other=0)
            correction = tl.dot(xh, wl, correction)
        piece = _rounding_boundary(primary + correction * (1. / 2048.))
        acc += piece * (1. / 256.)
    return acc


@triton.jit
def _zero_low_temporal(X0, X1, WH, WL, LowCount, Bias, Y,
                       M: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
                       T: tl.constexpr, S: tl.constexpr, HAS_BIAS: tl.constexpr,
                       IMPLICIT: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    mt, nt = tl.cdiv(M, 64), tl.cdiv(N, 64)
    group = pid // (GROUP * nt)
    first = group * GROUP
    count = tl.minimum(GROUP, mt - first)
    local = pid % (GROUP * nt)
    rows = (first + local % count) * 64 + tl.arange(0, 64)
    cols = (local // count) * 64 + tl.arange(0, 64)
    if tl.load(LowCount) != 0:
        if IMPLICIT:
            acc = _implicit_sum(X0, WH, WL, rows, cols, M, N, C, T, S, True)
        else:
            acc = _short_sum(X0, X1, WH, WL, rows, cols, M, N, C * 3, 64, True)
    else:
        if IMPLICIT:
            acc = _implicit_sum(X0, WH, WL, rows, cols, M, N, C, T, S, False)
        else:
            acc = _short_sum(X0, X1, WH, WL, rows, cols, M, N, C * 3, 64, False)
    if HAS_BIAS:
        acc += tl.load(Bias + cols, cols < N, other=0)[None, :]
    offset = ((rows[:, None] // (T * S)) * N + cols[None, :]) * (T * S) + rows[:, None] % (T * S)
    tl.store(Y + offset, acc, (rows[:, None] < M) & (cols[None, :] < N))


def zero_weight_low_temporal(value, parts, low_count, bias=None, *, implementation='auto'):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim != 5 or not value.is_contiguous()):
        raise ValueError('Expected contiguous inference CUDA FP32 NCTHW')
    batch, channels, frames, height, width = value.shape
    if len(parts) != 2 or any(p.ndim != 2 or p.shape != parts[0].shape
            or p.shape[1] != channels * 3 or p.dtype != torch.float16
            or p.device != value.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid temporal half weight parts')
    outputs = parts[0].shape[0]
    if min(batch, channels, frames, height, width, outputs) <= 0:
        raise ValueError('Empty temporal dimensions unsupported')
    if (low_count.shape != () or low_count.dtype != torch.int64
            or low_count.device != value.device):
        raise ValueError('Invalid device residual count')
    if bias is not None and (bias.shape != (outputs,) or bias.dtype != torch.float32
            or bias.device != value.device or not bias.is_contiguous()):
        raise ValueError('Invalid temporal FP32 bias')
    if implementation == 'auto':
        implementation = 'implicit' if channels in {320, 640} else 'packed'
    if implementation not in {'implicit', 'packed'}:
        raise ValueError('Unknown temporal implementation')
    import os
    mode=os.environ.get('UNIFOLM_WORLD_PACKED_TEMPORAL_SCHEDULE','0')
    if mode not in {'0','1'}:
        raise ValueError('Invalid packed temporal schedule')
    selected=(mode=='1' and implementation=='packed' and (batch,channels,frames,outputs)==(1,1280,16,1280)
        and (height,width) in {(10,16),(5,8)}
        and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
    rows, spatial = batch * frames * height * width, height * width
    if implementation == 'packed':
        packed = tuple(empty_conv_buffer('temporal', (rows, channels * 3),
            device=value.device, dtype=torch.float16) for _ in range(2))
        _pack_temporal[(triton.cdiv(rows, 32), triton.cdiv(channels * 3, 32))](
            value, *packed, rows, channels, frames, spatial, 32, 32,
            num_warps=4, enable_fp_fusion=False)
    else:
        packed = value, value
    output = empty_conv_buffer('temporal', (batch, outputs, frames, height, width),
                               device=value.device, dtype=torch.float32)
    group = (8 if channels == 320 else 16) if implementation == 'implicit' else 32
    if selected:group=8 if spatial==160 else 32
    stages=3 if selected else 2
    _zero_low_temporal[(triton.cdiv(rows, 64) * triton.cdiv(outputs, 64),)](
        *packed, *parts, low_count, parts[0] if bias is None else bias, output,
        rows, outputs, channels, frames, spatial, bias is not None,
        implementation == 'implicit', group, num_warps=4, num_stages=stages,
        enable_fp_fusion=False)
    return output
