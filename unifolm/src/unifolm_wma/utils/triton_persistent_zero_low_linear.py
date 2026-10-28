import torch
import triton
import triton.language as tl
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary

@triton.jit
def _short_sum_width(XH, XL, WH, WL, rows, cols, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, THREE: tl.constexpr, BN: tl.constexpr):
    inner = tl.arange(0, 64)
    accumulator = tl.full((BM, BN), 0.0, tl.float32)
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
        piece = _rounding_boundary(primary + correction * (1.0 / 2048.0))
        accumulator += piece * (1.0 / 256.0)
    return accumulator

@triton.jit
def _zero_low_gemm_persistent(XH, XL, WH, WL, LowCount, Bias, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr, BM: tl.constexpr, GROUP: tl.constexpr, OUTPUT_GROUPS: tl.constexpr, BN: tl.constexpr, THREE: tl.constexpr):
    if (tl.load(LowCount) != 0) == THREE:
        if THREE:
            row_tiles, col_tiles = (tl.cdiv(M, BM), tl.cdiv(N, BN))
            for pid in range(tl.program_id(0), row_tiles * col_tiles, tl.num_programs(0)):
                group = pid // (GROUP * col_tiles)
                first = group * GROUP
                count = tl.minimum(GROUP, row_tiles - first)
                local = pid % (GROUP * col_tiles)
                rows = (first + local % count) * BM + tl.arange(0, BM)
                cols = local // count * BN + tl.arange(0, BN)
                accumulator = _short_sum_width(XH, XL, WH, WL, rows, cols, M, N, K, BM, THREE, BN)
                if HAS_BIAS:
                    accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
                if OUTPUT_GROUPS == 1:
                    offset = rows[:, None] * N + cols[None, :]
                else:
                    width = N // OUTPUT_GROUPS
                    offset = cols[None, :] // width * M * width + rows[:, None] * width + cols[None, :] % width
                tl.store(Y + offset, accumulator, (rows[:, None] < M) & (cols[None, :] < N))
        else:
            pid = tl.program_id(0)
            row_tiles, col_tiles = (tl.cdiv(M, BM), tl.cdiv(N, BN))
            group = pid // (GROUP * col_tiles)
            first = group * GROUP
            count = tl.minimum(GROUP, row_tiles - first)
            local = pid % (GROUP * col_tiles)
            rows = (first + local % count) * BM + tl.arange(0, BM)
            cols = local // count * BN + tl.arange(0, BN)
            accumulator = _short_sum_width(XH, XL, WH, WL, rows, cols, M, N, K, BM, THREE, BN)
            if HAS_BIAS:
                accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
            if OUTPUT_GROUPS == 1:
                offset = rows[:, None] * N + cols[None, :]
            else:
                width = N // OUTPUT_GROUPS
                offset = cols[None, :] // width * M * width + rows[:, None] * width + cols[None, :] % width
            tl.store(Y + offset, accumulator, (rows[:, None] < M) & (cols[None, :] < N))

def launch_one(inputs, weights, count, bias, *, m, n, k, output_groups, config, three, output=None, inspect=False):
    assert not torch.is_grad_enabled() and (not torch.is_autocast_enabled())
    xh, xl = inputs
    wh, wl = weights
    device = xh.device
    assert all((p.is_cuda and p.dtype == torch.float16 and (p.device == device) and p.is_contiguous() for p in (*inputs, *weights)))
    assert xh.numel() == xl.numel() == m * k and wh.shape == wl.shape == (n, k)
    assert count.shape == () and count.dtype == torch.int64 and (count.device == device)
    assert output_groups in (1, 3) and n % output_groups == 0
    shape = (m, n) if output_groups == 1 else (output_groups, m, n // output_groups)
    if output is None:
        output = torch.empty(shape, device=device, dtype=torch.float32)
    assert output.numel() == m * n and output.is_contiguous() and (output.dtype == torch.float32) and (output.device == device)
    assert bias is None or (bias.shape == (n,) and bias.dtype == torch.float32 and (bias.device == device) and bias.is_contiguous())
    bm, bn, group, warps, stages = config
    compiled = _zero_low_gemm_persistent[min(triton.cdiv(m, bm) * triton.cdiv(n, bn), 216) if three else triton.cdiv(m, bm) * triton.cdiv(n, bn),](*inputs, *weights, count, wh if bias is None else bias, output, m, n, k, bias is not None, bm, group, output_groups, bn, three, num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return (output, compiled) if inspect else output

def fused(inputs, weights, count, bias, *, m, n, k, output_groups, config, fallback_config, output=None, inspect=False):
    output, first = launch_one(inputs, weights, count, bias, m=m, n=n, k=k, output_groups=output_groups, config=config, three=False, output=output, inspect=True)
    _, second = launch_one(inputs, weights, count, bias, m=m, n=n, k=k, output_groups=output_groups, config=fallback_config, three=True, output=output, inspect=True)
    return (output, (first, second)) if inspect else output
