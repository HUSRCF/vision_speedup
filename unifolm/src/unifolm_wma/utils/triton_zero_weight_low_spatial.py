"""Skip proven zero filter-residual products, retaining original short sums."""
import os
import torch
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary


def selected(kind):
    mode=os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL','0')
    if mode not in {'0','packed','f2','all'}:
        raise ValueError('Invalid spatial zero filter-residual scope')
    return mode in {kind,'all'}


def _signature(parts):
    try:
        return tuple((p.data_ptr(),p._version,tuple(p.shape),p.dtype,p.device) for p in parts)
    except RuntimeError:
        return None


def residual_counts(parts,kind,*,refresh=False):
    if not selected(kind):return None
    signature=_signature(parts)
    if signature is None:return None
    cached=getattr(parts[0],'_spatial_filter_residual_counts',None)
    if not refresh and cached is not None and cached[0]==signature and cached[1]==kind:
        return cached[2]
    if torch.cuda.is_current_stream_capturing():return None
    low=parts[1]
    count=(torch.count_nonzero(low,dim=(1,2)) if kind=='f2' else torch.count_nonzero(low))
    if cached is not None and cached[1]==kind and cached[2].shape==count.shape and cached[2].device==count.device:
        cached[2].copy_(count);count=cached[2]
    parts[0]._spatial_filter_residual_counts=(signature,kind,count)
    return count


@triton.jit
def _sum(XH,XL,WH,WL,rows,cols,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLITS:tl.constexpr,
         split,THREE:tl.constexpr):
    inner=tl.arange(0,BK)
    acc=tl.full((BM,BN),0.,tl.float32)
    steps=tl.cdiv(tl.cdiv(K,BK),SPLITS)
    for block in range(steps):
        k=(split*steps+block)*BK+inner
        xm=(rows[:,None]<M)&(k[None,:]<K)
        wm=(cols[None,:]<N)&(k[:,None]<K)
        xh=tl.load(XH+rows[:,None]*K+k[None,:],xm,other=0)
        xl=tl.load(XL+rows[:,None]*K+k[None,:],xm,other=0)
        wh=tl.load(WH+cols[None,:]*K+k[:,None],wm,other=0)
        primary=_rounding_boundary(tl.dot(xh,wh))
        correction=tl.dot(xl,wh)
        if THREE:
            wl=tl.load(WL+cols[None,:]*K+k[:,None],wm,other=0)
            correction=tl.dot(xh,wl,correction)
        acc+=_rounding_boundary(primary+correction*(1./2048.))*(1./256.)
    return acc


@triton.jit
def _packed_gemm(XH,XL,WH,WL,Counts,Bias,Output,M:tl.constexpr,N:tl.constexpr,
                 K:tl.constexpr,S:tl.constexpr,HAS_BIAS:tl.constexpr,
                 BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                 SPLITS:tl.constexpr,GROUP:tl.constexpr):
    pid,split=tl.program_id(0),tl.program_id(1)
    mt,nt=tl.cdiv(M,BM),tl.cdiv(N,BN)
    group=pid//(GROUP*nt);first=group*GROUP
    count=tl.minimum(GROUP,mt-first);local=pid%(GROUP*nt)
    rows=(first+local%count)*BM+tl.arange(0,BM)
    cols=(local//count)*BN+tl.arange(0,BN)
    if tl.load(Counts)!=0:
        acc=_sum(XH,XL,WH,WL,rows,cols,M,N,K,BM,BN,BK,SPLITS,split,True)
    else:
        acc=_sum(XH,XL,WH,WL,rows,cols,M,N,K,BM,BN,BK,SPLITS,split,False)
    if SPLITS==1:
        if HAS_BIAS:acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
        offset=((rows[:,None]//S)*N+cols[None,:])*S+rows[:,None]%S
    else:
        offset=split*M*N+rows[:,None]*N+cols[None,:]
    tl.store(Output+offset,acc,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def _f2_components(XH,XL,WH,WL,Counts,Output,M:tl.constexpr,N:tl.constexpr,
                    K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    part=tl.program_id(2)
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    xh,xl=XH+part*M*K,XL+part*M*K
    wh,wl=WH+part*N*K,WL+part*N*K
    if tl.load(Counts+part)!=0:
        acc=_sum(xh,xl,wh,wl,rows,cols,M,N,K,BM,BN,BK,1,0,True)
    else:
        acc=_sum(xh,xl,wh,wl,rows,cols,M,N,K,BM,BN,BK,1,0,False)
    tl.store(Output+part*M*N+rows[:,None]*N+cols[None,:],acc,
             (rows[:,None]<M)&(cols[None,:]<N))


def run_packed_components(packed,parts,bias,partial,rows,outputs,inner,spatial,
                          block_m,block_n,splits):
    counts=residual_counts(parts,'packed')
    if counts is None:return False
    _packed_gemm[(triton.cdiv(rows,block_m)*triton.cdiv(outputs,block_n),splits)](
        *packed,*parts,counts,parts[0] if bias is None else bias,partial,
        rows,outputs,inner,spatial,bias is not None,block_m,block_n,64,splits,16,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return True


def run_f2_components(packed,parts,product,rows,outputs,inputs,block_m,block_n,
                      block_k,num_warps,num_stages):
    counts=residual_counts(parts,'f2')
    if counts is None:return False
    _f2_components[(triton.cdiv(rows,block_m),triton.cdiv(outputs,block_n),16)](
        *packed,*parts,counts,product,rows,outputs,inputs,block_m,block_n,block_k,
        num_warps=num_warps,num_stages=num_stages,enable_fp_fusion=False)
    return True
