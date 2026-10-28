import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import _half_split
from .triton_zero_weight_low_geglu import _sum_geglu

@triton.jit
def _zero_geglu_hidden_parts(XH,XL,WH,WL,Count,Bias,YH,YL,V,G,M:tl.constexpr,C:tl.constexpr,K:tl.constexpr,
                HAS_BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                GROUP:tl.constexpr,PRIMARY_VALUE:tl.constexpr,PRIMARY_GATE:tl.constexpr,
                DEBUG:tl.constexpr):
    pid=tl.program_id(0)
    mt,nt=tl.cdiv(M,BM),tl.cdiv(C,BN)
    group=pid//(GROUP*nt);first=group*GROUP
    count=tl.minimum(GROUP,mt-first);local=pid%(GROUP*nt)
    rows=(first+local%count)*BM+tl.arange(0,BM)
    cols=(local//count)*BN+tl.arange(0,BN)
    if tl.load(Count)!=0:
        value,gate=_sum_geglu(XH,XL,WH,WL,rows,cols,M,C,K,BM,BN,BK,PRIMARY_VALUE,PRIMARY_GATE,True)
    else:
        value,gate=_sum_geglu(XH,XL,WH,WL,rows,cols,M,C,K,BM,BN,BK,PRIMARY_VALUE,PRIMARY_GATE,False)
    if PRIMARY_VALUE:value*=1./256.
    if PRIMARY_GATE:gate*=1./256.
    if HAS_BIAS:
        value+=tl.load(Bias+cols,cols<C,other=0)[None,:]
        gate+=tl.load(Bias+cols+C,cols<C,other=0)[None,:]
    offset=rows[:,None]*C+cols[None,:]
    valid=(rows[:,None]<M)&(cols[None,:]<C)
    if DEBUG:
        tl.store(V+offset,value,valid);tl.store(G+offset,gate,valid)
    gelu=(gate*.5)*(1.+tl.math.erf(gate*.7071067811865476))
    hidden=_rounding_boundary(value*gelu)
    high,low=_half_split(hidden,1.)
    store_offset=tl.inline_asm_elementwise("mov.b32 $0, $1;",constraints="=r,r",
        args=[offset],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(YH+store_offset,high,valid)
    tl.store(YL+store_offset,low,valid)


@triton.jit
def _packed_two_product_gemm_residual(XH, XL, High, Low, Bias, Residual, Output,
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
        x_high=tl.load(XH+rows[:,None]*K+k[None,:],
                       (rows[:,None]<M)&(k[None,:]<K),other=0)
        if INPUT_PROTECTED:
            x_low=tl.load(XL+rows[:,None]*K+k[None,:],
                          (rows[:,None]<M)&(k[None,:]<K),other=0)
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

