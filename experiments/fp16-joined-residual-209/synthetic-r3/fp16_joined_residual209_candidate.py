"""Experimental two-part Half product in one physical K128 MMA.

This changes summation order and is not an exact replacement. Logical K64
outer FP32 accumulation is retained. Static weights with nonzero low parts
take the original three-product path. No input/output cache is used.
"""
import triton
import triton.language as tl
from unifolm_wma.utils.triton_zero_weight_low_linear import _short_sum
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary


@triton.jit
def _joined(XH, XL, WH, WL, LowCount, Bias, Residual, Y,
            M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            HAS_BIAS: tl.constexpr, BM: tl.constexpr, A_SHIFT: tl.constexpr):
    pid = tl.program_id(0)
    rt, ct = tl.cdiv(M, BM), tl.cdiv(N, 64)
    first = (pid // (32 * ct)) * 32
    count = tl.minimum(32, rt - first)
    local = pid % (32 * ct)
    rows = (first + local % count) * BM + tl.arange(0, BM)
    cols = (local // count) * 64 + tl.arange(0, 64)
    if tl.load(LowCount) != 0:
        acc = _short_sum(XH, XL, WH, WL, rows, cols, M, N, K, BM, True)
    else:
        acc = tl.full((BM, 64), 0., tl.float32)
        inner = tl.arange(0, 128)
        second = inner >= 64
        xp = tl.where(second, XL, XH)
        xs = tl.where(second, 1. / (2 ** A_SHIFT), 1.)
        ws = tl.where(second, 1. / (2 ** (11 - A_SHIFT)), 1.)
        for block in range(tl.cdiv(K, 64)):
            ks = block * 64 + inner % 64
            x = tl.load(xp[None, :] + rows[:, None] * K + ks[None, :],
                        (rows[:, None] < M) & (ks[None, :] < K), other=0)
            w = tl.load(WH + cols[None, :] * K + ks[:, None],
                        (cols[None, :] < N) & (ks[:, None] < K), other=0)
            a = (x.to(tl.float32) * xs[None, :]).to(tl.float16)
            b = (w.to(tl.float32) * ws[:, None]).to(tl.float16)
            acc += _rounding_boundary(tl.dot(a, b)) * (1. / 256.)
    if HAS_BIAS:
        acc += tl.load(Bias + cols, cols < N, other=0)[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    acc += tl.load(Residual + rows[:, None] * N + cols[None, :], mask, other=0)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, mask)


def launch(values, out, *, shift, bm):
    m, n, k = (values[name] for name in ('M', 'N', 'K'))
    assert 0 <= shift <= 11 and bm in (32, 64)
    return _joined[(triton.cdiv(m, bm) * triton.cdiv(n, 64),)](
        *(values[name] for name in ('XH', 'XL', 'WH', 'WL', 'LowCount', 'Bias', 'Residual')),
        out, m, n, k, values['HAS_BIAS'], bm, shift,
        **values['_launch_options'])


@triton.jit
def _fused(XH, XL, WH, WL, LowCount, Bias, Residual, Y,
           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           HAS_BIAS: tl.constexpr, BM: tl.constexpr, A_SHIFT: tl.constexpr):
    pid = tl.program_id(0)
    rt, ct = tl.cdiv(M, BM), tl.cdiv(N, 64)
    first = (pid // (32 * ct)) * 32
    count = tl.minimum(32, rt - first)
    local = pid % (32 * ct)
    rows = (first + local % count) * BM + tl.arange(0, BM)
    cols = (local // count) * 64 + tl.arange(0, 64)
    if tl.load(LowCount) != 0:
        acc = _short_sum(XH, XL, WH, WL, rows, cols, M, N, K, BM, True)
    else:
        acc = tl.full((BM, 64), 0., tl.float32)
        inner = tl.arange(0, 64)
        for block in range(tl.cdiv(K, 64)):
            ks = block * 64 + inner
            xm = (rows[:, None] < M) & (ks[None, :] < K)
            wm = (cols[None, :] < N) & (ks[:, None] < K)
            xh = tl.load(XH + rows[:, None] * K + ks[None, :], xm, other=0)
            xl = tl.load(XL + rows[:, None] * K + ks[None, :], xm, other=0)
            wh = tl.load(WH + cols[None, :] * K + ks[:, None], wm, other=0)
            xc = (xl.to(tl.float32) * (1. / (2 ** A_SHIFT))).to(tl.float16)
            if A_SHIFT == 11:
                wc = wh
            else:
                wc = (wh.to(tl.float32) * (1. / (2 ** (11 - A_SHIFT)))).to(tl.float16)
            primary = _rounding_boundary(tl.dot(xh, wh))
            piece = tl.dot(xc, wc, primary)
            acc += _rounding_boundary(piece) * (1. / 256.)
    if HAS_BIAS:
        acc += tl.load(Bias + cols, cols < N, other=0)[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    acc += tl.load(Residual + rows[:, None] * N + cols[None, :], mask, other=0)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, mask)


def launch_fused(values, out, *, shift, bm, method):
    m, n, k = (values[name] for name in ('M', 'N', 'K'))
    assert 0 <= shift <= 11 and bm in (32, 64) and method == 'chained'
    return _fused[(triton.cdiv(m, bm) * triton.cdiv(n, 64),)](
        *(values[name] for name in ('XH', 'XL', 'WH', 'WL', 'LowCount', 'Bias', 'Residual')),
        out, m, n, k, values['HAS_BIAS'], bm, shift,
        **values['_launch_options'])
