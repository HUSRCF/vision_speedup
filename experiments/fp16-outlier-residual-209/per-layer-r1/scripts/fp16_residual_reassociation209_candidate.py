"""An exact-Half residual control to isolate FP32 summation reordering."""
from fp16_outlier_packed_residual209_candidate import *
from unifolm_wma.utils.triton_zero_weight_low_linear import _short_sum
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary


@triton.jit
def _separate(XH,XL,WH,WL,Count,Bias,Residual,Y,
              M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,BM:tl.constexpr):
    pid=tl.program_id(0);rt=tl.cdiv(M,BM);ct=tl.cdiv(N,64)
    first=(pid//(32*ct))*32;count=tl.minimum(32,rt-first);local=pid%(32*ct)
    rows=(first+local%count)*BM+tl.arange(0,BM)
    cols=(local//count)*64+tl.arange(0,64)
    if tl.load(Count)!=0:
        acc=_short_sum(XH,XL,WH,WL,rows,cols,M,N,K,BM,True)
    else:
        primary_sum=tl.full((BM,64),0.,tl.float32)
        correction_sum=tl.full((BM,64),0.,tl.float32)
        for block in range(tl.cdiv(K,64)):
            ks=block*64+tl.arange(0,64)
            maskx=(rows[:,None]<M)&(ks[None,:]<K)
            maskw=(cols[None,:]<N)&(ks[:,None]<K)
            xh=tl.load(XH+rows[:,None]*K+ks[None,:],maskx,other=0)
            xl=tl.load(XL+rows[:,None]*K+ks[None,:],maskx,other=0)
            wh=tl.load(WH+cols[None,:]*K+ks[:,None],maskw,other=0)
            primary_sum+=_rounding_boundary(tl.dot(xh,wh))*(1./256.)
            correction_sum+=_rounding_boundary(tl.dot(xl,wh))*(1./(2048.*256.))
        acc=primary_sum+correction_sum
    if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
    mask=(rows[:,None]<M)&(cols[None,:]<N)
    acc+=tl.load(Residual+rows[:,None]*N+cols[None,:],mask,other=0)
    tl.store(Y+rows[:,None]*N+cols[None,:],acc,mask)


def launch_separate(values,out):
    m,n,k=(values[name] for name in ('M','N','K'));bm=values['BM']
    return _separate[(triton.cdiv(m,bm)*triton.cdiv(n,64),)](
        *(values[name] for name in ('XH','XL','WH','WL','LowCount','Bias','Residual')),
        out,m,n,k,values['HAS_BIAS'],bm,**values['_launch_options'])
