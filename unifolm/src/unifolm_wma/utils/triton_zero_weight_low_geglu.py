"""Device-predicated zero residual path with original GEGLU tiling and sums."""
import os
import torch
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary


def selected(inner):
    mode=os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU','0')
    if mode not in {'0','c320','c640','c1280','all'}:
        raise ValueError('Invalid GEGLU zero filter-residual scope')
    return (not torch.backends.cudnn.allow_tf32 and inner in {320,640,1280}
            and mode in {'all','c'+str(inner)})


def residual_count(parts,*,refresh=False):
    if not selected(parts[0].shape[1]):return None
    try:
        signature=tuple((p.data_ptr(),p._version,tuple(p.shape),p.dtype,p.device) for p in parts)
    except RuntimeError:return None
    cached=getattr(parts[0],'_geglu_zero_weight_low_count',None)
    if not refresh and cached is not None and cached[0]==signature:return cached[1]
    if torch.cuda.is_current_stream_capturing():return None
    count=torch.count_nonzero(parts[1])
    if cached is not None and cached[1].device==count.device:
        cached[1].copy_(count);count=cached[1]
    parts[0]._geglu_zero_weight_low_count=(signature,count)
    return count


@triton.jit
def _short_product(xh,xl,wh,wl,THREE:tl.constexpr):
    primary=_rounding_boundary(tl.dot(xh,wh))
    correction=tl.dot(xl,wh)
    if THREE:correction=tl.dot(xh,wl,correction)
    return _rounding_boundary(primary+correction*(1./2048.))*(1./256.)


@triton.jit
def _sum_geglu(XH,XL,WH,WL,rows,cols,M:tl.constexpr,C:tl.constexpr,K:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
               PRIMARY_VALUE:tl.constexpr,PRIMARY_GATE:tl.constexpr,THREE:tl.constexpr):
    inner=tl.arange(0,BK)
    value=tl.full((BM,BN),0.,tl.float32)
    gate=tl.full((BM,BN),0.,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=block*BK+inner
        xo=rows[:,None]*K+k[None,:]
        xv=(rows[:,None]<M)&(k[None,:]<K)
        xh=tl.load(XH+xo,xv,other=0)
        if not (PRIMARY_VALUE and PRIMARY_GATE):xl=tl.load(XL+xo,xv,other=0)
        wo=cols[None,:]*K+k[:,None]
        wv=(cols[None,:]<C)&(k[:,None]<K)
        vh=tl.load(WH+wo,wv,other=0)
        gh=tl.load(WH+wo+C*K,wv,other=0)
        if PRIMARY_VALUE:
            value=tl.dot(xh,vh,value)
        else:
            vl=vh
            if THREE:vl=tl.load(WL+wo,wv,other=0)
            value+=_short_product(xh,xl,vh,vl,THREE)
        if PRIMARY_GATE:
            gate=tl.dot(xh,gh,gate)
        else:
            gl=gh
            if THREE:gl=tl.load(WL+wo+C*K,wv,other=0)
            gate+=_short_product(xh,xl,gh,gl,THREE)
    return value,gate


@triton.jit
def _zero_geglu(XH,XL,WH,WL,Count,Bias,Y,V,G,M:tl.constexpr,C:tl.constexpr,K:tl.constexpr,
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
    tl.store(Y+offset,value*gelu,valid)


def run(packed,parts,bias,output,v,g,rows,channels,inner,block_m,block_n,group,
        primary_value,primary_gate,debug,block_k,num_warps,num_stages):
    if primary_value and primary_gate:return False
    count=residual_count(parts)
    if count is None:return False
    _zero_geglu[(triton.cdiv(rows,block_m)*triton.cdiv(channels,block_n),)](
        *packed,*parts,count,parts[0] if bias is None else bias,output,v,g,
        rows,channels,inner,bias is not None,block_m,block_n,block_k,group,
        primary_value,primary_gate,debug,num_warps=num_warps,num_stages=num_stages,
        enable_fp_fusion=False)
    return True
