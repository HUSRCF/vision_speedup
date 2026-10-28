import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import _half_split
from .triton_zero_weight_low_geglu import _sum_geglu

@triton.jit
def _zero_geglu_hidden_parts_vector_store(XH,XL,WH,WL,Count,Bias,YH,YL,V,G,M:tl.constexpr,C:tl.constexpr,K:tl.constexpr,
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
    store_offset=offset
    tl.store(YH+store_offset,high,valid)
    tl.store(YL+store_offset,low,valid)

