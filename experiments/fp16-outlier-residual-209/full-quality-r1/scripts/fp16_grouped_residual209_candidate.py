"""Accumulate INT8 corrections across larger groups before dequantization."""
import torch
import triton
import triton.language as tl
import fp16_scalar_residual209_candidate as previous
from fp16_scalar_residual209_candidate import allocate_pack,pack,pack_channels,launch
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary


@triton.jit
def _pack_group(X,Q,S,R:tl.constexpr,K:tl.constexpr,GK:tl.constexpr,P:tl.constexpr):
    row=tl.program_id(0);group=tl.program_id(1)
    index=tl.arange(0,P);ks=group*GK+index
    active=(ks<K)&(index<GK)
    value=tl.load(X+row*K+ks,active,other=0).to(tl.float32)
    maximum=tl.max(tl.abs(value),0)
    scale=tl.where(maximum>0,maximum/127.,1.)
    q=tl.inline_asm_elementwise('cvt.rni.s32.f32 $0, $1;',
        constraints='=r,f',args=[value/scale],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(Q+row*K+ks,q,active)
    tl.store(S+row*tl.cdiv(K,GK)+group,scale)


@triton.jit
def _group_body(XH,WH,XQ,WQ,XS,WS,rows,cols,
                M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,GK:tl.constexpr):
    inner=tl.arange(0,64)
    primary_sum=tl.full((BM,BN),0.,tl.float32)
    correction_sum=tl.full((BM,BN),0.,tl.float32)
    for group in range(tl.cdiv(K,GK)):
        correction=tl.full((BM,BN),0,tl.int32)
        blocks=tl.minimum(tl.cdiv(GK,64),tl.cdiv(K-group*GK,64))
        for block in range(blocks):
            ks=group*GK+block*64+inner
            xo=rows[:,None]*K+ks[None,:]
            wo=cols[None,:]*K+ks[:,None]
            xm=(rows[:,None]<M)&(ks[None,:]<K)
            wm=(cols[None,:]<N)&(ks[:,None]<K)
            xh=tl.load(XH+xo,xm,other=0);wh=tl.load(WH+wo,wm,other=0)
            primary=_rounding_boundary(tl.dot(xh,wh))
            primary_sum+=primary*(1./256.)
            xq=tl.load(XQ+xo,xm,other=0);wq=tl.load(WQ+wo,wm,other=0)
            correction=tl.dot(xq,wq,correction)
        sx=tl.load(XS+rows*tl.cdiv(K,GK)+group,rows<M,other=0)
        sw=tl.load(WS+cols*tl.cdiv(K,GK)+group,cols<N,other=0)
        correction_sum+=correction.to(tl.float32)*(sx[:,None]*sw[None,:])*(1./(2048.*256.))
    return primary_sum+correction_sum


@triton.jit
def _group_gemm(XH,XL,WH,WL,Count,XQ,WQ,XS,WS,Bias,Residual,Y,
                M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,GK:tl.constexpr):
    pid=tl.program_id(0)
    row_tiles=tl.cdiv(M,BM);col_tiles=tl.cdiv(N,BN)
    first=(pid//(32*col_tiles))*32
    count=tl.minimum(32,row_tiles-first);local=pid%(32*col_tiles)
    rows=(first+local%count)*BM+tl.arange(0,BM)
    cols=(local//count)*BN+tl.arange(0,BN)
    if tl.load(Count)!=0:
        acc=previous._body(XH,XL,WH,WL,XQ,WQ,XS,WS,rows,cols,M,N,K,BM,BN,1,True)
    else:
        acc=_group_body(XH,WH,XQ,WQ,XS,WS,rows,cols,M,N,K,BM,BN,GK)
    if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
    mask=(rows[:,None]<M)&(cols[None,:]<N)
    acc+=tl.load(Residual+rows[:,None]*N+cols[None,:],mask,other=0)
    tl.store(Y+rows[:,None]*N+cols[None,:],acc,mask)


def allocate_group_pack(x,gk):
    assert x.is_cuda and x.ndim==2 and x.is_contiguous() and gk>0 and gk%64==0
    return (torch.empty_like(x,dtype=torch.int8),
            torch.empty((x.shape[0],triton.cdiv(x.shape[1],gk)),device=x.device,dtype=torch.float32))


def pack_group(x,out,gk):
    r,k=x.shape
    _pack_group[(r,triton.cdiv(k,gk))](x,*out,r,k,gk,triton.next_power_of_2(gk),
        num_warps=4,enable_fp_fusion=False)


def launch_group(values,xpack,wpack,out,*,bn,gk):
    m,n,k=(values[name] for name in ('M','N','K'));bm=64 if m>=2560 else 32
    return _group_gemm[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
        *(values[name] for name in ('XH','XL','WH','WL','LowCount')),
        xpack[0],wpack[0],xpack[1],wpack[1],
        *(values[name] for name in ('Bias','Residual')),out,
        m,n,k,values['HAS_BIAS'],bm,bn,gk,
        num_warps=4,num_stages=values.get('_candidate_stages',2),enable_fp_fusion=False)
