"""Keep logical BK64 rounding while loading/MMA in two ordered K32 slices."""
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import _half_split


@triton.jit
def _subtile32_sum(XH,XL,WH,WL,rows,cols,M:tl.constexpr,C:tl.constexpr,K:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,THREE:tl.constexpr):
    inner=tl.arange(0,32)
    value=tl.full((BM,BN),0.,tl.float32)
    gate=tl.full((BM,BN),0.,tl.float32)
    for block in range(tl.cdiv(K,64)):
        primary=tl.full((BM,BN),0.,tl.float32)
        correction=tl.full((BM,BN),0.,tl.float32)
        for sub in tl.static_range(0,2):
            k=block*64+sub*32+inner
            xo=rows[:,None]*K+k[None,:]
            xv=(rows[:,None]<M)&(k[None,:]<K)
            xh=tl.load(XH+xo,xv,other=0)
            xl=tl.load(XL+xo,xv,other=0)
            wo=cols[None,:]*K+k[:,None]
            wv=(cols[None,:]<C)&(k[:,None]<K)
            vh=tl.load(WH+wo,wv,other=0)
            gh=tl.load(WH+wo+C*K,wv,other=0)
            primary=tl.dot(xh,vh,primary)
            correction=tl.dot(xl,vh,correction)
            gate=tl.dot(xh,gh,gate)
        # Original correction sums all XL*WH before any XH*WL; interleaving
        # these terms would change FP32 arithmetic and is deliberately avoided.
        if THREE:
            for sub in tl.static_range(0,2):
                k=block*64+sub*32+inner
                xo=rows[:,None]*K+k[None,:]
                xv=(rows[:,None]<M)&(k[None,:]<K)
                xh=tl.load(XH+xo,xv,other=0)
                wo=cols[None,:]*K+k[:,None]
                wv=(cols[None,:]<C)&(k[:,None]<K)
                vl=tl.load(WL+wo,wv,other=0)
                correction=tl.dot(xh,vl,correction)
        primary=_rounding_boundary(primary)
        short=_rounding_boundary(primary+correction*(1./2048.))*(1./256.)
        value+=short
    return value,gate


@triton.jit
def _geglu_subtile32(XH,XL,WH,WL,Count,Bias,YH,YL,V,G,M:tl.constexpr,C:tl.constexpr,K:tl.constexpr,
                     HAS_BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                     GROUP:tl.constexpr,PRIMARY_VALUE:tl.constexpr,PRIMARY_GATE:tl.constexpr,
                     DEBUG:tl.constexpr):
    tl.static_assert(not PRIMARY_VALUE and PRIMARY_GATE)
    tl.static_assert(BK==64)
    pid=tl.program_id(0)
    mt,nt=tl.cdiv(M,BM),tl.cdiv(C,BN)
    group=pid//(GROUP*nt);first=group*GROUP
    count=tl.minimum(GROUP,mt-first);local=pid%(GROUP*nt)
    rows=(first+local%count)*BM+tl.arange(0,BM)
    cols=(local//count)*BN+tl.arange(0,BN)
    if tl.load(Count)!=0:
        value,gate=_subtile32_sum(XH,XL,WH,WL,rows,cols,M,C,K,BM,BN,True)
    else:
        value,gate=_subtile32_sum(XH,XL,WH,WL,rows,cols,M,C,K,BM,BN,False)
    gate*=1./256.
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
    tl.store(YH+offset,high,valid)
    tl.store(YL+offset,low,valid)
