"""Primitive-only FP16/scalar and FP16 plus quantized residual candidates.

No production installation. Keep the original K64 FP32 rounding boundaries.
Only the lower-significance correction product may use INT8 Tensor Cores.
"""
import torch
import triton
import triton.language as tl
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary


@triton.jit
def _pack64(X, Low, Q, S, R: tl.constexpr, K: tl.constexpr,
            MODE: tl.constexpr, TARGET: tl.constexpr, POWER2: tl.constexpr,
            PARTS:tl.constexpr):
    rows=tl.program_id(0)*4+tl.arange(0,4)
    group=tl.program_id(1)
    ks=group*64+tl.arange(0,64)
    value=tl.load(X+rows[:,None]*K+ks[None,:],
                  (rows[:,None]<R)&(ks[None,:]<K),other=0).to(tl.float32)
    if PARTS:
        value+=tl.load(Low+rows[:,None]*K+ks[None,:],
            (rows[:,None]<R)&(ks[None,:]<K),other=0).to(tl.float32)*(1./2048.)
    maximum=tl.max(tl.abs(value),1)
    scale=tl.where(maximum>0,maximum/TARGET,1.)
    if POWER2:scale=tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))
    normalized=value/scale[:,None]
    if MODE==0:
        normalized=tl.inline_asm_elementwise('cvt.rni.s32.f32 $0, $1;',
            constraints='=r,f',args=[normalized],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(Q+rows[:,None]*K+ks[None,:],normalized,
             (rows[:,None]<R)&(ks[None,:]<K))
    tl.store(S+rows*tl.cdiv(K,64)+group,scale,rows<R)


@triton.jit
def _body(XH,XL,WH,WL,XQ,WQ,XS,WS,rows,cols,
          M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,MODE:tl.constexpr,THREE:tl.constexpr):
    inner=tl.arange(0,64)
    acc=tl.full((BM,BN),0.,tl.float32)
    for block in range(tl.cdiv(K,64)):
        ks=block*64+inner
        xo=rows[:,None]*K+ks[None,:]
        wo=cols[None,:]*K+ks[:,None]
        xm=(rows[:,None]<M)&(ks[None,:]<K)
        wm=(cols[None,:]<N)&(ks[:,None]<K)
        if MODE==3:
            xh=tl.load(XQ+xo,xm,other=0)
            wh=tl.load(WQ+wo,wm,other=0)
        else:
            xh=tl.load(XH+xo,xm,other=0)
            wh=tl.load(WH+wo,wm,other=0)
        primary=_rounding_boundary(tl.dot(xh,wh))
        if MODE==1 or MODE==2:
            if MODE==2:
                xq=tl.load(XQ+xo,xm,other=0)
                wq=tl.load(WQ+wo,wm,other=0)
                correction=tl.dot(xq,wq).to(tl.float32)
                sx=tl.load(XS+rows*tl.cdiv(K,64)+block,rows<M,other=0)
                sw=tl.load(WS+cols*tl.cdiv(K,64)+block,cols<N,other=0)
                correction=correction*(sx[:,None]*sw[None,:])
            else:
                xl=tl.load(XL+xo,xm,other=0)
                correction=tl.dot(xl,wh)
                if THREE:
                    wl=tl.load(WL+wo,wm,other=0)
                    correction=tl.dot(xh,wl,correction)
            piece=_rounding_boundary(primary+correction*(1./2048.))
        elif MODE==3:
            sx=tl.load(XS+rows*tl.cdiv(K,64)+block,rows<M,other=0)
            sw=tl.load(WS+cols*tl.cdiv(K,64)+block,cols<N,other=0)
            piece=primary*(sx[:,None]*sw[None,:])
        else:piece=primary
        acc+=piece*(1./256.)
    return acc


@triton.jit
def _gemm(XH, XL, WH, WL, Count, XQ, WQ, XS, WS, Bias, Residual, Y,
          M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,MODE:tl.constexpr):
    # Hoist the uniform weight predicate outside K, like the production
    # kernel. Each compiled loop has one statically fixed MMA combination.
    pid=tl.program_id(0)
    row_tiles=tl.cdiv(M,BM);col_tiles=tl.cdiv(N,BN)
    first=(pid//(32*col_tiles))*32
    count=tl.minimum(32,row_tiles-first)
    local=pid%(32*col_tiles)
    rows=(first+local%count)*BM+tl.arange(0,BM)
    cols=(local//count)*BN+tl.arange(0,BN)
    if MODE==1 or MODE==2:
        if tl.load(Count)!=0:
            acc=_body(XH,XL,WH,WL,XQ,WQ,XS,WS,rows,cols,M,N,K,BM,BN,1,True)
        else:
            acc=_body(XH,XL,WH,WL,XQ,WQ,XS,WS,rows,cols,M,N,K,BM,BN,MODE,False)
    else:
        acc=_body(XH,XL,WH,WL,XQ,WQ,XS,WS,rows,cols,M,N,K,BM,BN,MODE,False)
    if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
    acc+=tl.load(Residual+rows[:,None]*N+cols[None,:],
                 (rows[:,None]<M)&(cols[None,:]<N),other=0)
    tl.store(Y+rows[:,None]*N+cols[None,:],acc,
             (rows[:,None]<M)&(cols[None,:]<N))


def allocate_pack(x,*,half=False):
    assert x.is_cuda and x.ndim==2 and x.is_contiguous()
    return (torch.empty_like(x,dtype=torch.float16 if half else torch.int8),
            torch.empty((x.shape[0],triton.cdiv(x.shape[1],64)),
                        device=x.device,dtype=torch.float32))


def pack(x,out,*,half=False,power2=False,target=None,low=None):
    r,k=x.shape
    _pack64[(triton.cdiv(r,4),triton.cdiv(k,64))](
        x,x if low is None else low,*out,r,k,int(half),
        float(target or (256 if half else 127)),power2,low is not None,
        num_warps=4,enable_fp_fusion=False)
    return out


def launch(values,xpack,wpack,out,*,mode,bn=64):
    m,n,k=values['M'],values['N'],values['K'];bm=64 if m>=2560 else 32
    return _gemm[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
        *(values[n] for n in ('XH','XL','WH','WL','LowCount')),
        xpack[0],wpack[0],xpack[1],wpack[1],
        *(values[n] for n in ('Bias','Residual')),out,
        m,n,k,values['HAS_BIAS'],bm,bn,mode,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
