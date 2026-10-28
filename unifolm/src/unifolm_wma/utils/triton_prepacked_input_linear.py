"""Per-call input splitting for the accepted FP32-output compensated GEMM.

Input parts are always refreshed in this call, including CUDA Graph replay.
Only static weight parts may be supplied by the caller's validated warm cache.
"""
import os
import torch
from .gemm_output_allocation import empty as empty_gemm_output
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import pack_half_weight


@triton.jit
def _prepacked_gemm(XH, XL, WH, WL, Bias, Output,
                     M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr, GROUP_M: tl.constexpr,
                     OUTPUT_GROUPS: tl.constexpr = 1):
    if GROUP_M:
        pid = tl.program_id(0)
        row_tiles = tl.cdiv(M, BM)
        col_tiles = tl.cdiv(N, BN)
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
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0., tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        input_offset = rows[:, None] * K + k[None, :]
        input_valid = (rows[:, None] < M) & (k[None, :] < K)
        xh = tl.load(XH + input_offset, input_valid, other=0)
        xl = tl.load(XL + input_offset, input_valid, other=0)
        weight_offset = cols[None, :] * K + k[:, None]
        weight_valid = (cols[None, :] < N) & (k[:, None] < K)
        wh = tl.load(WH + weight_offset, weight_valid, other=0)
        wl = tl.load(WL + weight_offset, weight_valid, other=0)
        primary = _rounding_boundary(tl.dot(xh, wh))
        correction = tl.dot(xl, wh)
        correction = tl.dot(xh, wl, correction)
        block = _rounding_boundary(primary + correction * (1. / 2048.))
        accumulator += block * (1. / 256.)
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    if OUTPUT_GROUPS == 1:
        output_offset = rows[:, None] * N + cols[None, :]
    else:
        part_n = N // OUTPUT_GROUPS
        output_offset = ((cols[None, :] // part_n) * M * part_n +
                         rows[:, None] * part_n + cols[None, :] % part_n)
    tl.store(Output + output_offset, accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))


def input_pipeline_stages(rows, requested=None):
    if requested is not None:
        return requested
    mode = os.environ.get('UNIFOLM_COMPENSATED_LINEAR_INPUT_PIPELINE', '2')
    if mode not in {'2', '3'}:
        raise ValueError('Input pipeline must be 2 or 3')
    return 3 if mode == '3' and rows >= 2560 else 2


def prepacked_input_linear(value, weight, bias=None, *, weight_parts=None,
                            block_m=32, block_n=64, block_k=64, group_m=32,
                            return_kernel=False, num_warps=4, num_stages=None,
                            output_groups=1):
    if (not value.is_cuda or value.dtype != torch.float32 or value.ndim < 1
            or not value.is_contiguous() or weight.ndim != 2
            or weight.dtype != torch.float32 or not weight.is_contiguous()
            or value.device != weight.device or value.shape[-1] != weight.shape[1]
            or torch.is_grad_enabled()):
        raise ValueError('Requires contiguous CUDA FP32 inference input and weight')
    rows = value.numel() // value.shape[-1]
    num_stages = input_pipeline_stages(rows, num_stages)
    if (block_m not in {32, 64, 128} or block_n not in {64, 128}
            or block_k not in {32, 64, 128} or group_m not in {0, 4, 8, 16, 32}
            or num_warps not in {4, 8} or num_stages not in {2, 3}):
        raise ValueError('Unsupported prepacked-input GEMM configuration')
    channels, reduction = weight.shape
    if output_groups not in {1, 3} or channels % output_groups:
        raise ValueError('Output groups must be 1 or 3 and divide output channels')
    if min(rows, channels, reduction) <= 0:
        raise ValueError('Empty dimensions are unsupported')
    if bias is not None and (bias.shape != (channels,) or bias.device != value.device
                            or bias.dtype != torch.float32 or not bias.is_contiguous()):
        raise ValueError('Requires contiguous FP32 bias')
    parts = pack_half_weight(weight) if weight_parts is None else weight_parts
    if len(parts) != 2 or any(p.shape != weight.shape or p.device != value.device
                             or p.dtype != torch.float16 or not p.is_contiguous()
                             for p in parts):
        raise ValueError('Invalid compensated weight parts')
    # This conversion kernel is intentionally included in every call. Its
    # outputs must never be reused across distinct diffusion iterations.
    xh, xl = pack_half_weight(value.view(rows, reduction), weight_scale=1.)
    output_shape = ((*value.shape[:-1], channels) if output_groups == 1 else
                    (output_groups, *value.shape[:-1], channels // output_groups))
    output = empty_gemm_output('classic',output_shape, device=value.device,
                         dtype=torch.float32)
    row_tiles, col_tiles = triton.cdiv(rows, block_m), triton.cdiv(channels, block_n)
    grid = (row_tiles * col_tiles,) if group_m else (row_tiles, col_tiles)
    kernel = _prepacked_gemm[grid](xh, xl, *parts,
        weight if bias is None else bias, output, rows, channels, reduction,
        bias is not None, block_m, block_n, block_k, group_m, output_groups,
        num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False)
    return (output, kernel) if return_kernel else output


from .decision_zero_low_runtime import wrap_gemm as _decision_zero_low_wrap_gemm
_prepacked_gemm=_decision_zero_low_wrap_gemm(_prepacked_gemm)


from .world_zero_low_runtime import wrap_gemm as _world_zero_low_wrap_gemm
_prepacked_gemm=_world_zero_low_wrap_gemm(_prepacked_gemm)
