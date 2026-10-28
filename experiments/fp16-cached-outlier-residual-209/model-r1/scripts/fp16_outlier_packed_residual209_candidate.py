"""Pack protected residual channels densely inside each INT8 pack kernel.

The main Half product and its K64 FP32 summation stay unchanged. This is
approximate arithmetic: matching a local error gate never certifies video PSNR.
"""
import triton
import triton.language as tl
import torch
from fp16_balanced_residual209_candidate import allocate_group_pack as _allocate
from fp16_balanced_residual209_candidate import allocate_pack,pack,pack_channels,launch
from fp16_grouped_residual209_split_candidate import _fallback
from fp16_grouped_residual209_candidate import _group_body
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary


@triton.jit
def _pack(X,Channel,Q,S,Dense,Outlier,R:tl.constexpr,K:tl.constexpr,
          GK:tl.constexpr,P:tl.constexpr,INVERSE:tl.constexpr,EXACT_K:tl.constexpr,PE:tl.constexpr):
    row=tl.program_id(0);group=tl.program_id(1)
    index=tl.arange(0,P);ks=group*GK+index
    mask=(ks<K)&(index<GK)
    value=tl.load(X+row*K+ks,mask,other=0).to(tl.float32)
    channel=tl.load(Channel+ks,mask,other=1.)
    # A zero channel marks an exact-Half outlier. Exclude it from both the
    # quantized values and scale estimation without ever dividing by zero.
    safe=tl.where(channel>0,channel,1.)
    if INVERSE:value=value/safe
    else:value=value*safe
    value=tl.where(channel>0,value,0.)
    maximum=tl.max(tl.abs(value),0)
    scale=tl.where(maximum>0,maximum/127.,1.)
    quantized=tl.inline_asm_elementwise('cvt.rni.s32.f32 $0, $1;',
        constraints='=r,f',args=[value/scale],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(Q+row*K+ks,quantized,mask)
    tl.store(S+row*tl.cdiv(K,GK)+group,scale)
    if EXACT_K>0:
        if group==0:
            oi=tl.arange(0,PE)
            exact_indices=tl.load(Outlier+oi,oi<EXACT_K,other=0)
            exact_value=tl.load(X+row*K+exact_indices,oi<EXACT_K,other=0)
            tl.store(Dense+row*EXACT_K+oi,exact_value,oi<EXACT_K)


@triton.jit
def _fast(XH,XL,WH,Count,XQ,WQ,XS,WS,XD,WD,Bias,Residual,Y,
          M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,GK:tl.constexpr,EXACT_K:tl.constexpr):
    if tl.load(Count)==0:
        pid=tl.program_id(0);rt=tl.cdiv(M,BM);ct=tl.cdiv(N,BN)
        first=(pid//(32*ct))*32;count=tl.minimum(32,rt-first);local=pid%(32*ct)
        rows=(first+local%count)*BM+tl.arange(0,BM)
        cols=(local//count)*BN+tl.arange(0,BN)
        acc=_group_body(XH,WH,XQ,WQ,XS,WS,rows,cols,M,N,K,BM,BN,GK)
        if EXACT_K>0:
            exact=tl.full((BM,BN),0.,tl.float32)
            for block in range(tl.cdiv(EXACT_K,32)):
                oi=block*32+tl.arange(0,32)
                xl=tl.load(XD+rows[:,None]*EXACT_K+oi[None,:],
                    (rows[:,None]<M)&(oi[None,:]<EXACT_K),other=0)
                wh=tl.load(WD+cols[None,:]*EXACT_K+oi[:,None],
                    (cols[None,:]<N)&(oi[:,None]<EXACT_K),other=0)
                exact=tl.dot(xl,wh,exact)
            acc+=_rounding_boundary(exact)*(1./(2048.*256.))
        if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
        mask=(rows[:,None]<M)&(cols[None,:]<N)
        acc+=tl.load(Residual+rows[:,None]*N+cols[None,:],mask,other=0)
        tl.store(Y+rows[:,None]*N+cols[None,:],acc,mask)


def pack_balanced(value,out,gk,channel,*,inverse=False,outliers):
    rows,k=value.shape
    assert channel.shape==(k,) and channel.is_contiguous()
    assert out[2].shape==(rows,outliers.numel())
    _pack[(rows,triton.cdiv(k,gk))](value,channel,*out,outliers,rows,k,gk,
        triton.next_power_of_2(gk),inverse,outliers.numel(),
        triton.next_power_of_2(max(1,outliers.numel())),num_warps=4,enable_fp_fusion=False)


def launch_group(values,xpack,wpack,out,*,bn,gk):
    m,n,k=(values[name] for name in ('M','N','K'));bm=values['_candidate_bm']
    indices=values['_outlier_indices']
    fast=_fast[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
        values['XH'],values['XL'],values['WH'],values['LowCount'],
        xpack[0],wpack[0],xpack[1],wpack[1],xpack[2],wpack[2],
        values['Bias'],values['Residual'],out,m,n,k,values['HAS_BIAS'],bm,bn,gk,
        indices.numel(),num_warps=4,num_stages=values['_candidate_stages'],enable_fp_fusion=False)
    original_bm=values['BM']
    _fallback[(triton.cdiv(m,original_bm)*triton.cdiv(n,64),)](
        *(values[name] for name in ('XH','XL','WH','WL','LowCount','Bias','Residual')),out,
        m,n,k,values['HAS_BIAS'],original_bm,
        num_warps=4,num_stages=values['_launch_options']['num_stages'],enable_fp_fusion=False)
    return fast


def allocate_group_pack(value,gk,exact_k=0):
    ordinary=_allocate(value,gk)
    return (*ordinary,torch.empty((value.shape[0],exact_k),device=value.device,dtype=torch.float16))
