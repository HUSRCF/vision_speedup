"""Skip zero weight-residual products with a graph-visible device predicate.

Every BK64 reduction boundary stays unchanged. The predicate is refreshed with
cached weight parts; nonzero residuals execute the full three-product path.
"""
import torch; import os
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import pack_half_weight
from .gemm_output_allocation import empty as empty_gemm_output


def weight_low_count(parts, *, out=None):
    if len(parts) != 2 or not parts[1].is_cuda or parts[1].dtype != torch.float16:
        raise ValueError('Expected CUDA half weight parts')
    result = torch.empty((), device=parts[1].device, dtype=torch.int64) if out is None else out
    if result.shape != () or result.device != parts[1].device or result.dtype != torch.int64:
        raise ValueError('Invalid persistent weight residual count')
    result.copy_(torch.count_nonzero(parts[1]))
    return result


@triton.jit
def _short_sum(XH, XL, WH, WL, rows, cols,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, THREE: tl.constexpr):
    inner = tl.arange(0, 64)
    accumulator = tl.full((BM, 64), 0., tl.float32)
    for block in range(tl.cdiv(K, 64)):
        k = block * 64 + inner
        xo = rows[:, None] * K + k[None, :]
        xv = (rows[:, None] < M) & (k[None, :] < K)
        xh = tl.load(XH + xo, xv, other=0)
        xl = tl.load(XL + xo, xv, other=0)
        wo = cols[None, :] * K + k[:, None]
        wv = (cols[None, :] < N) & (k[:, None] < K)
        wh = tl.load(WH + wo, wv, other=0)
        primary = _rounding_boundary(tl.dot(xh, wh))
        correction = tl.dot(xl, wh)
        if THREE:
            wl = tl.load(WL + wo, wv, other=0)
            correction = tl.dot(xh, wl, correction)
        piece = _rounding_boundary(primary + correction * (1. / 2048.))
        accumulator += piece * (1. / 256.)
    return accumulator


@triton.jit
def _zero_low_gemm(XH, XL, WH, WL, LowCount, Bias, Y,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   HAS_BIAS: tl.constexpr, BM: tl.constexpr, GROUP: tl.constexpr,
                   OUTPUT_GROUPS: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(N, 64)
    group = pid // (GROUP * col_tiles)
    first = group * GROUP
    count = tl.minimum(GROUP, row_tiles - first)
    local = pid % (GROUP * col_tiles)
    rows = (first + local % count) * BM + tl.arange(0, BM)
    cols = (local // count) * 64 + tl.arange(0, 64)
    # Uniform dispatch once per tile; both inner loops have static MMA counts.
    if tl.load(LowCount) != 0:
        accumulator = _short_sum(XH, XL, WH, WL, rows, cols, M, N, K, BM, True)
    else:
        accumulator = _short_sum(XH, XL, WH, WL, rows, cols, M, N, K, BM, False)
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    if OUTPUT_GROUPS == 1:
        offset = rows[:, None] * N + cols[None, :]
    else:
        width = N // OUTPUT_GROUPS
        offset = ((cols[None, :] // width) * M * width +
                  rows[:, None] * width + cols[None, :] % width)
    tl.store(Y + offset, accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))


def zero_weight_low_linear(value, parts, low_count, bias=None, *, debug_poison=False,
                           output_groups=1):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim < 1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous inference CUDA FP32 input')
    if len(parts) != 2 or any(p.ndim != 2 or p.shape != parts[0].shape
            or p.shape[1] != value.shape[-1] or p.device != value.device
            or p.dtype != torch.float16 or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid half weight parts')
    if (low_count.shape != () or low_count.device != value.device
            or low_count.dtype != torch.int64):
        raise ValueError('Invalid device weight-residual count')
    rows = value.numel() // value.shape[-1]
    channels, inner = parts[0].shape
    if output_groups not in {1, 3} or channels % output_groups:
        raise ValueError('Output groups must be 1 or 3 and divide channels')
    if min(rows, channels, inner) <= 0:
        raise ValueError('Empty shapes unsupported')
    if bias is not None and (bias.shape != (channels,) or bias.device != value.device
            or bias.dtype != torch.float32 or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    packed = pack_half_weight(value.view(rows, inner), weight_scale=1.)
    shape = ((*value.shape[:-1], channels) if output_groups == 1 else
             (output_groups, *value.shape[:-1], channels // output_groups))
    output = empty_gemm_output('classic', shape,
                              device=value.device, dtype=torch.float32)
    if debug_poison:
        output.fill_(float('nan'))
    bm = 64 if rows >= 2560 else 32
    qkv_stages=os.environ.get('UNIFOLM_WORLD_ZERO_QKV_STAGES','2')
    if qkv_stages not in {'2','3-large'}:raise ValueError('Invalid zero-low QKV schedule')
    large_qkv={(40960,960,320),(10240,1920,640),(2560,3840,1280)}
    stages=(3 if qkv_stages=='3-large' and output_groups==3 and bias is None
        and (rows,channels,inner) in large_qkv
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cuda.matmul.allow_tf32 else 2)
    qkv_group=os.environ.get('UNIFOLM_WORLD_ZERO_QKV_GROUP','32')
    if qkv_group not in {'32','1-large'}:
        raise ValueError('Invalid zero-low QKV group')
    selected_group=(1 if qkv_group=='1-large' and output_groups==3 and bias is None
        and (rows,channels,inner) in large_qkv
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cuda.matmul.allow_tf32 else 32)
    _zero_low_gemm[(triton.cdiv(rows, bm) * triton.cdiv(channels, 64),)](
        *packed, *parts, low_count, parts[0] if bias is None else bias, output,
        rows, channels, inner, bias is not None, bm, selected_group, output_groups,
        num_warps=4, num_stages=stages, enable_fp_fusion=False)
    return output


from .world_zero_low_stages_runtime import wrap_gemm as _protected_scores_wrap
_zero_low_gemm=_protected_scores_wrap(_zero_low_gemm)


from .decision_zero_low_stages_runtime import wrap_gemm as _decision_stages_wrap
_zero_low_gemm=_decision_stages_wrap(_zero_low_gemm)
