"""Experimental FP32 compensated GEMM plus GEGLU in one kernel.

The two halves keep independent short-block FP32 accumulators. Only the
activated 1280-channel result is written to memory; no projected intermediate
is materialized. Requires full-video validation before production use.
"""
import os
import torch
import triton
import triton.language as tl
from unifolm_wma.utils.triton_compensated_linear import _split, _rounding_boundary


@triton.jit
def _compensated_geglu(X, Weight, Bias, Output,
                       M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                       HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr,
                       GROUP_M: tl.constexpr = 0):
    if GROUP_M:
        pid = tl.program_id(0)
        row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(C, BN)
        group = pid // (GROUP_M * col_tiles)
        first_row = group * GROUP_M
        group_rows = tl.minimum(GROUP_M, row_tiles-first_row)
        local = pid % (GROUP_M * col_tiles)
        row_tile = first_row + local % group_rows
        col_tile = local // group_rows
    else:
        row_tile, col_tile = tl.program_id(0), tl.program_id(1)
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    value_acc = tl.full((BM, BN), 0, tl.float32)
    gate_acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        value_w = tl.load(Weight + cols[None, :] * K + k[:, None],
                          (cols[None, :] < C) & (k[:, None] < K), other=0)
        gate_w = tl.load(Weight + (cols[None, :] + C) * K + k[:, None],
                         (cols[None, :] < C) & (k[:, None] < K), other=0)
        x_high, x_low = _split(x)
        vw_high, vw_low = _split(value_w)
        gw_high, gw_low = _split(gate_w)
        vp = _rounding_boundary(tl.dot(x_high, vw_high, allow_tf32=True))
        vc = tl.dot(x_low, vw_high, allow_tf32=True)
        vc = tl.dot(x_high, vw_low, vc, allow_tf32=True)
        gp = _rounding_boundary(tl.dot(x_high, gw_high, allow_tf32=True))
        gc = tl.dot(x_low, gw_high, allow_tf32=True)
        gc = tl.dot(x_high, gw_low, gc, allow_tf32=True)
        value_acc += _rounding_boundary(vp + vc)
        gate_acc += _rounding_boundary(gp + gc)
    if HAS_BIAS:
        value_acc += tl.load(Bias + cols, cols < C, other=0)[None, :]
        gate_acc += tl.load(Bias + C + cols, cols < C, other=0)[None, :]
    gelu = (gate_acc * 0.5) * (1. + tl.math.erf(gate_acc * 0.7071067811865476))
    result = value_acc * gelu
    tl.store(Output + rows[:, None] * C + cols[None, :], result,
             (rows[:, None] < M) & (cols[None, :] < C))


def compensated_geglu(value, weight, bias=None, *, block_m=64, block_n=32,
                      num_warps=4, num_stages=2, group_m=0):
    if group_m not in {0, 4, 8, 16, 32}:
        raise ValueError('Unsupported GEGLU row tile group')
    if (not value.is_cuda or value.dtype != torch.float32 or
            weight.dtype != torch.float32 or value.device != weight.device or
            not value.is_contiguous() or not weight.is_contiguous() or
            value.shape[-1] != weight.shape[1] or weight.shape[0] % 2 or
            torch.is_grad_enabled()):
        raise ValueError('Fused compensated GEGLU requires contiguous CUDA FP32 inference')
    rows = value.numel() // value.shape[-1]
    channels, reduction = weight.shape[0] // 2, weight.shape[1]
    output = torch.empty((*value.shape[:-1], channels), device=value.device,
                         dtype=torch.float32)
    row_tiles, col_tiles = triton.cdiv(rows, block_m), triton.cdiv(channels, block_n)
    grid = (row_tiles*col_tiles,) if group_m else (row_tiles, col_tiles)
    _compensated_geglu[grid](
        value, weight, weight if bias is None else bias, output,
        rows, channels, reduction, bias is not None,
        BM=block_m, BN=block_n, BK=32, GROUP_M=group_m, num_warps=num_warps,
        num_stages=num_stages, enable_fp_fusion=False)
    return output


def make_forward(module):
    original = module.forward
    projection = module.proj
    group_m = int(os.environ.get('UNIFOLM_COMPENSATED_GEGLU_GROUP_M', '0'))
    if group_m not in {0, 4, 8, 16, 32}:
        raise ValueError('Unsupported GEGLU row tile group')

    def forward(value):
        if (module.training or torch.is_grad_enabled() or not value.is_cuda or
                value.dtype != torch.float32 or projection.weight.dtype != torch.float32 or
                not value.is_contiguous() or not projection.weight.is_contiguous() or
                projection.in_features != 320 or projection.out_features != 2560 or
                value.numel() // projection.in_features != 40960):
            return original(value)
        return compensated_geglu(value, projection.weight, projection.bias, group_m=group_m)

    return forward


def install_compensated_geglus(model):
    from unifolm_wma.modules.attention import GEGLU
    count = 0
    for module in model.modules():
        if (isinstance(module, GEGLU) and module.proj.in_features == 320 and
                module.proj.out_features == 2560):
            if getattr(module, '_compensated_geglu_installed', False):
                raise RuntimeError('Compensated GEGLU already installed')
            module.forward = make_forward(module)
            module._compensated_geglu_installed = True
            count += 1
    return count
