"""Experimental FP16 delta GEMM; fresh inputs, cached pre-bias FP32 product.

Full refresh keeps the original BK64 sum. Reuse approximates only the input
difference. This is not a production owner and requires explicit refresh after
weight/condition changes. Full video quality remains an independent gate.
"""
import torch
import triton
import triton.language as tl
from unifolm_wma.utils.triton_zero_weight_low_linear import _short_sum
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary

@triton.jit
def _save(XH,XL,Previous,Size:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<Size
    value=tl.load(XH+i,mask,other=0).to(tl.float32)+tl.load(XL+i,mask,other=0).to(tl.float32)*(1./2048.)
    tl.store(Previous+i,value,mask)

@triton.jit
def _pack_delta(XH,XL,Previous,Delta,Size:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<Size
    value=tl.load(XH+i,mask,other=0).to(tl.float32)+tl.load(XL+i,mask,other=0).to(tl.float32)*(1./2048.)
    delta=(value-tl.load(Previous+i,mask,other=0))*.25
    # Fixed power-of-two scale keeps differences of all finite Half pairs in
    # Half range. Normal values retain the usual Half relative precision.
    tl.store(Delta+i,delta.to(tl.float16),mask)

@triton.jit
def _refresh(XH,XL,WH,WL,Count,Bias,Residual,Y,Cached,
             M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,
             BM:tl.constexpr,GROUP:tl.constexpr):
    pid=tl.program_id(0);mt,nt=tl.cdiv(M,BM),tl.cdiv(N,64)
    group=pid//(GROUP*nt);first=group*GROUP;count=tl.minimum(GROUP,mt-first);local=pid%(GROUP*nt)
    rows=(first+local%count)*BM+tl.arange(0,BM);cols=(local//count)*64+tl.arange(0,64)
    if tl.load(Count)!=0:acc=_short_sum(XH,XL,WH,WL,rows,cols,M,N,K,BM,True)
    else:acc=_short_sum(XH,XL,WH,WL,rows,cols,M,N,K,BM,False)
    offsets=rows[:,None]*N+cols[None,:];mask=(rows[:,None]<M)&(cols[None,:]<N)
    tl.store(Cached+offsets,acc,mask)
    if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
    acc+=tl.load(Residual+offsets,mask,other=0);tl.store(Y+offsets,acc,mask)

@triton.jit
def _delta_sum(Delta,WH,WL,rows,cols,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
               BM:tl.constexpr,THREE:tl.constexpr):
    inner=tl.arange(0,64);acc=tl.full((BM,64),0.,tl.float32)
    for block in range(tl.cdiv(K,64)):
        k=block*64+inner
        d=tl.load(Delta+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        mask=(cols[None,:]<N)&(k[:,None]<K);offset=cols[None,:]*K+k[:,None]
        w=tl.load(WH+offset,mask,other=0);piece=_rounding_boundary(tl.dot(d,w))
        if THREE:
            low=tl.load(WL+offset,mask,other=0)
            piece=_rounding_boundary(piece+tl.dot(d,low)*(1./2048.))
        acc+=piece*(4./256.)
    return acc

@triton.jit
def _reuse(Delta,WH,WL,Count,Bias,Residual,Y,Cached,
           M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,
           BM:tl.constexpr,GROUP:tl.constexpr):
    pid=tl.program_id(0);mt,nt=tl.cdiv(M,BM),tl.cdiv(N,64)
    group=pid//(GROUP*nt);first=group*GROUP;count=tl.minimum(GROUP,mt-first);local=pid%(GROUP*nt)
    rows=(first+local%count)*BM+tl.arange(0,BM);cols=(local//count)*64+tl.arange(0,64)
    if tl.load(Count)!=0:acc=_delta_sum(Delta,WH,WL,rows,cols,M,N,K,BM,True)
    else:acc=_delta_sum(Delta,WH,WL,rows,cols,M,N,K,BM,False)
    offsets=rows[:,None]*N+cols[None,:];mask=(rows[:,None]<M)&(cols[None,:]<N)
    acc=tl.load(Cached+offsets,mask,other=0)+acc
    if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
    acc+=tl.load(Residual+offsets,mask,other=0);tl.store(Y+offsets,acc,mask)

def allocate(high,channels):
    m,k=high.shape
    return dict(previous=torch.empty((m,k),device=high.device,dtype=torch.float32),delta=torch.empty_like(high),product=torch.empty((m,channels),device=high.device,dtype=torch.float32))

@triton.jit
def _state_pack(XH,XL,Previous,Delta,Reuse,Size:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<Size
    value=tl.load(XH+i,mask,other=0).to(tl.float32)+tl.load(XL+i,mask,other=0).to(tl.float32)*(1./2048.)
    if tl.load(Reuse)!=0:
        delta=(value-tl.load(Previous+i,mask,other=0))*.25
        tl.store(Delta+i,delta.to(tl.float16),mask)
    else:tl.store(Previous+i,value,mask)

@triton.jit
def _incremental(XH,XL,WH,WL,Count,Bias,Residual,Y,Cached,Delta,Reuse,
                 M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,
                 BM:tl.constexpr,GROUP:tl.constexpr):
    pid=tl.program_id(0);mt,nt=tl.cdiv(M,BM),tl.cdiv(N,64)
    group=pid//(GROUP*nt);first=group*GROUP;count=tl.minimum(GROUP,mt-first);local=pid%(GROUP*nt)
    rows=(first+local%count)*BM+tl.arange(0,BM);cols=(local//count)*64+tl.arange(0,64)
    offsets=rows[:,None]*N+cols[None,:];mask=(rows[:,None]<M)&(cols[None,:]<N)
    if tl.load(Reuse)!=0:
        if tl.load(Count)!=0:acc=_delta_sum(Delta,WH,WL,rows,cols,M,N,K,BM,True)
        else:acc=_delta_sum(Delta,WH,WL,rows,cols,M,N,K,BM,False)
        acc=tl.load(Cached+offsets,mask,other=0)+acc
    else:
        if tl.load(Count)!=0:acc=_short_sum(XH,XL,WH,WL,rows,cols,M,N,K,BM,True)
        else:acc=_short_sum(XH,XL,WH,WL,rows,cols,M,N,K,BM,False)
        tl.store(Cached+offsets,acc,mask)
    if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
    acc+=tl.load(Residual+offsets,mask,other=0);tl.store(Y+offsets,acc,mask)

def run_device(inputs,parts,count,bias,residual,output,cache,flag,*,bm=64,stages=3):
    high,low=inputs;wh,wl=parts;m,k=high.shape;n=wh.shape[0]
    assert flag.shape==()and flag.dtype==torch.int32 and flag.device==high.device
    _state_pack[(triton.cdiv(m*k,1024),)](high,low,cache['previous'],cache['delta'],flag,m*k,1024,num_warps=4,enable_fp_fusion=False)
    _incremental[(triton.cdiv(m,bm)*triton.cdiv(n,64),)](high,low,wh,wl,count,wh if bias is None else bias,residual,output,cache['product'],cache['delta'],flag,m,n,k,bias is not None,bm,32,num_warps=4,num_stages=stages,enable_fp_fusion=False)
    return output

def run(inputs,parts,count,bias,residual,output,cache,*,refresh,bm=64,stages=3):
    high,low=inputs;wh,wl=parts;m,k=high.shape;n=wh.shape[0]
    assert high.is_cuda and not torch.is_grad_enabled() and high.dtype==low.dtype==wh.dtype==wl.dtype==torch.float16
    assert high.shape==low.shape==(m,k)and wh.shape==wl.shape==(n,k)and residual.shape==output.shape==(m,n)
    assert all(x.is_contiguous()for x in (high,low,wh,wl,residual,output))
    grid=(triton.cdiv(m,bm)*triton.cdiv(n,64),)
    b=wh if bias is None else bias
    if refresh:
        _refresh[grid](high,low,wh,wl,count,b,residual,output,cache['product'],m,n,k,bias is not None,bm,32,num_warps=4,num_stages=stages,enable_fp_fusion=False)
        _save[(triton.cdiv(m*k,1024),)](high,low,cache['previous'],m*k,1024,num_warps=4,enable_fp_fusion=False)
    else:
        _pack_delta[(triton.cdiv(m*k,1024),)](high,low,cache['previous'],cache['delta'],m*k,1024,num_warps=4,enable_fp_fusion=False)
        _reuse[grid](cache['delta'],wh,wl,count,b,residual,output,cache['product'],m,n,k,bias is not None,bm,32,num_warps=4,num_stages=stages,enable_fp_fusion=False)
    return output
