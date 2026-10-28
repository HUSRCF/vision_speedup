import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary

@triton.jit
def _prepacked_gemm_residual(XH, XL, WH, WL, Bias, Residual, Output,
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
    accumulator += tl.load(Residual + rows[:, None] * N + cols[None, :],
                           (rows[:, None] < M) & (cols[None, :] < N), other=0)
    tl.store(Output + output_offset, accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))

