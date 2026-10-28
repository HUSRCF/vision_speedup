import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import _half_split


@triton.jit
def _half_compensated_gemm_residual(X, Weight, WeightLow, Bias, Residual, Output,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                           BN: tl.constexpr, XS: tl.constexpr, WS: tl.constexpr,
                           PACKED: tl.constexpr, GROUP_M: tl.constexpr = 0,
                           BK: tl.constexpr = 32):
    if GROUP_M:
        pid = tl.program_id(0)
        row_tiles = tl.cdiv(M, BM)
        col_tiles = tl.cdiv(N, BN)
        group = pid // (GROUP_M * col_tiles)
        first_row = group * GROUP_M
        group_rows = tl.minimum(GROUP_M, row_tiles - first_row)
        within_group = pid % (GROUP_M * col_tiles)
        row_tile = first_row + within_group % group_rows
        col_tile = within_group // group_rows
    else:
        row_tile = tl.program_id(0)
        col_tile = tl.program_id(1)
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        x_high, x_low = _half_split(x, XS)
        weight = tl.load(Weight + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
        if PACKED:
            w_high = weight
            w_low = tl.load(WeightLow + cols[None, :] * K + k[:, None],
                            (cols[None, :] < N) & (k[:, None] < K), other=0)
        else:
            w_high, w_low = _half_split(weight, WS)
        primary = _rounding_boundary(tl.dot(x_high, w_high))
        correction = tl.dot(x_low, w_high)
        correction = tl.dot(x_high, w_low, correction)
        block = _rounding_boundary(primary + correction * (1 / 2048.0))
        accumulator += block * (1 / (XS * WS))
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    accumulator += tl.load(Residual + rows[:, None] * N + cols[None, :],
                           (rows[:, None] < M) & (cols[None, :] < N), other=0)
    tl.store(Output + rows[:, None] * N + cols[None, :], accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))

