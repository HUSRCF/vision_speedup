import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import _half_split


@triton.jit
def _two_product_gemm_residual(X, High, Low, Bias, Residual, Output,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      HAS_BIAS: tl.constexpr, INPUT_PROTECTED: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles = tl.cdiv(M, BM)
    col_tiles = tl.cdiv(N, BN)
    group = pid // (GROUP_M * col_tiles)
    first_row = group * GROUP_M
    group_rows = tl.minimum(GROUP_M, row_tiles - first_row)
    within_group = pid % (GROUP_M * col_tiles)
    rows = (first_row + within_group % group_rows) * BM + tl.arange(0, BM)
    cols = (within_group // group_rows) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        if INPUT_PROTECTED:
            x_high, x_low = _half_split(x, 1.0)
        else:
            x_high = x.to(tl.float16)
        w_high = tl.load(High + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
        primary = _rounding_boundary(tl.dot(x_high, w_high))
        if INPUT_PROTECTED:
            correction = tl.dot(x_low, w_high)
        else:
            w_low = tl.load(Low + cols[None, :] * K + k[:, None],
                            (cols[None, :] < N) & (k[:, None] < K), other=0)
            correction = tl.dot(x_high, w_low)
        block = _rounding_boundary(primary + correction * (1 / 2048.0))
        accumulator += block * (1 / 256.0)
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    accumulator += tl.load(Residual + rows[:, None] * N + cols[None, :],
                           (rows[:, None] < M) & (cols[None, :] < N), other=0)
    tl.store(Output + rows[:, None] * N + cols[None, :], accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))

