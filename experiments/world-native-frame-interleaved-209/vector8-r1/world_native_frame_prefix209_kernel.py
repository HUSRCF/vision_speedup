import triton as tr
import triton.language as tl
from unifolm_wma.utils.native_winograd_gemm_fp16_kernel import boundary as packed_boundary
from unifolm_wma.utils.native_winograd_gemm_fp16_kahan_kernel import boundary as kahan_boundary


@tr.jit
def transformed_gemm_fp16_packed_frame_prefix(WH, WL, X, Y, ROWS: tl.constexpr, COLS: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ACTIVE_COLS: tl.constexpr):
    tl.static_assert(ACTIVE_COLS * 2 == COLS)
    row_tile = tl.program_id(0) // tl.cdiv(ACTIVE_COLS, BN)
    col_tile = tl.program_id(0) % tl.cdiv(ACTIVE_COLS, BN)
    batch = tl.program_id(1)
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + inner
        offset = batch * ROWS * K + rows[:, None] * K + k[None, :]
        mask = (rows[:, None] < ROWS) & (k[None, :] < K)
        wh = tl.load(WH + offset, mask, other=0.0)
        wl = tl.load(WL + offset, mask, other=0.0)
        x = tl.load(X + batch * K * COLS + k[:, None] * COLS + cols[None, :], (k[:, None] < K) & (cols[None, :] < COLS), other=0.0)
        xh = x.to(tl.float16)
        xl = ((x - xh.to(tl.float32)) * 2048.0).to(tl.float16)
        primary = packed_boundary(tl.dot(wh, xh, allow_tf32=False))
        correction = tl.dot(wl, xh, allow_tf32=False)
        correction = tl.dot(wh, xl, correction, allow_tf32=False)
        acc += packed_boundary(primary + correction * (1.0 / 2048.0))
    tl.store(Y + batch * ROWS * COLS + rows[:, None] * COLS + cols[None, :], acc, (rows[:, None] < ROWS) & (cols[None, :] < COLS))

@tr.jit
def transformed_gemm_fp16_kahan_frame_prefix(W, X, Y, ROWS: tl.constexpr, COLS: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ACTIVE_COLS: tl.constexpr):
    tl.static_assert(ACTIVE_COLS * 2 == COLS)
    row_tile = tl.program_id(0) // tl.cdiv(ACTIVE_COLS, BN)
    col_tile = tl.program_id(0) % tl.cdiv(ACTIVE_COLS, BN)
    batch = tl.program_id(1)
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    compensation = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + inner
        w = tl.load(W + batch * ROWS * K + rows[:, None] * K + k[None, :], (rows[:, None] < ROWS) & (k[None, :] < K), other=0.0)
        x = tl.load(X + batch * K * COLS + k[:, None] * COLS + cols[None, :], (k[:, None] < K) & (cols[None, :] < COLS), other=0.0)
        wh = w.to(tl.float16)
        xh = x.to(tl.float16)
        wl = ((w - wh.to(tl.float32)) * 2048.0).to(tl.float16)
        xl = ((x - xh.to(tl.float32)) * 2048.0).to(tl.float16)
        primary = kahan_boundary(tl.dot(wh, xh, allow_tf32=False))
        correction = tl.dot(wl, xh, allow_tf32=False)
        correction = tl.dot(wh, xl, correction, allow_tf32=False)
        term = kahan_boundary(primary + correction * (1.0 / 2048.0))
        adjusted = kahan_boundary(term - compensation)
        updated = kahan_boundary(acc + adjusted)
        compensation = kahan_boundary(kahan_boundary(updated - acc) - adjusted)
        acc = updated
    tl.store(Y + batch * ROWS * COLS + rows[:, None] * COLS + cols[None, :], acc, (rows[:, None] < ROWS) & (cols[None, :] < COLS))
