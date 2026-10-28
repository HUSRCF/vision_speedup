"""Separate INT8 fast-path and original Half fallback device predicates."""
import torch
import triton
import triton.language as tl
import fp16_grouped_residual209_candidate as grouped
from fp16_grouped_residual209_candidate import allocate_group_pack,pack_group,allocate_pack,pack,pack_channels,launch


@triton.jit
def _fast(XH,WH,Count,XQ,WQ,XS,WS,Bias,Residual,Y,
          M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,GK:tl.constexpr):
    if tl.load(Count)==0:
        pid=tl.program_id(0);rt=tl.cdiv(M,BM);ct=tl.cdiv(N,BN)
        first=(pid//(32*ct))*32;count=tl.minimum(32,rt-first);local=pid%(32*ct)
        rows=(first+local%count)*BM+tl.arange(0,BM)
        cols=(local//count)*BN+tl.arange(0,BN)
        acc=grouped._group_body(XH,WH,XQ,WQ,XS,WS,rows,cols,M,N,K,BM,BN,GK)
        if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
        mask=(rows[:,None]<M)&(cols[None,:]<N)
        acc+=tl.load(Residual+rows[:,None]*N+cols[None,:],mask,other=0)
        tl.store(Y+rows[:,None]*N+cols[None,:],acc,mask)


@triton.jit
def _fallback(XH,XL,WH,WL,Count,Bias,Residual,Y,
              M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,
              BM:tl.constexpr):
    if tl.load(Count)!=0:
        pid=tl.program_id(0);rt=tl.cdiv(M,BM);ct=tl.cdiv(N,64)
        first=(pid//(32*ct))*32;count=tl.minimum(32,rt-first);local=pid%(32*ct)
        rows=(first+local%count)*BM+tl.arange(0,BM)
        cols=(local//count)*64+tl.arange(0,64)
        # Dummy quantized pointers/scales are not read by the compile-time
        # original three-Half-product branch.
        acc=grouped.previous._body(XH,XL,WH,WL,XL,WH,XL,WH,rows,cols,M,N,K,BM,64,1,True)
        if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
        mask=(rows[:,None]<M)&(cols[None,:]<N)
        acc+=tl.load(Residual+rows[:,None]*N+cols[None,:],mask,other=0)
        tl.store(Y+rows[:,None]*N+cols[None,:],acc,mask)


def launch_group(values,xpack,wpack,out,*,bn,gk):
    m,n,k=(values[name] for name in ('M','N','K'));bm=values['_candidate_bm']
    fast=_fast[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
        values['XH'],values['WH'],values['LowCount'],xpack[0],wpack[0],xpack[1],wpack[1],
        values['Bias'],values['Residual'],out,m,n,k,values['HAS_BIAS'],bm,bn,gk,
        num_warps=4,num_stages=values['_candidate_stages'],enable_fp_fusion=False)
    original_bm=values['BM']
    _fallback[(triton.cdiv(m,original_bm)*triton.cdiv(n,64),)](
        *(values[name] for name in ('XH','XL','WH','WL','LowCount','Bias','Residual')),out,
        m,n,k,values['HAS_BIAS'],original_bm,
        num_warps=4,num_stages=values['_launch_options']['num_stages'],enable_fp_fusion=False)
    return fast
