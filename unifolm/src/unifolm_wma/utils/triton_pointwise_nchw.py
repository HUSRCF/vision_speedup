"""Diagnostic direct-layout 1x1 convolution with short three-Half products."""
import torch
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary


@triton.jit
def _pointwise_nchw(X,WH,WL,Bias,Y,M:tl.constexpr,N:tl.constexpr,
                    K:tl.constexpr,S:tl.constexpr,HAS_BIAS:tl.constexpr,
                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                    GROUP_M:tl.constexpr):
    pid=tl.program_id(0)
    row_tiles,col_tiles=tl.cdiv(M,BM),tl.cdiv(N,BN)
    group=pid//(GROUP_M*col_tiles)
    first_row=group*GROUP_M
    group_rows=tl.minimum(GROUP_M,row_tiles-first_row)
    local=pid%(GROUP_M*col_tiles)
    rows=(first_row+local%group_rows)*BM+tl.arange(0,BM)
    cols=(local//group_rows)*BN+tl.arange(0,BN)
    inner=tl.arange(0,BK)
    batch=rows//S
    spatial=rows%S
    acc=tl.full((BM,BN),0.,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        k=start*BK+inner
        x=tl.load(X+(batch[:,None]*K+k[None,:])*S+spatial[:,None],
                  (rows[:,None]<M)&(k[None,:]<K),other=0.)
        xh,xl=_half_split(x,1.)
        offset=cols[None,:]*K+k[:,None]
        valid=(cols[None,:]<N)&(k[:,None]<K)
        wh=tl.load(WH+offset,valid,other=0.)
        wl=tl.load(WL+offset,valid,other=0.)
        primary=_rounding_boundary(tl.dot(xh,wh))
        correction=tl.dot(xl,wh)
        correction=tl.dot(xh,wl,correction)
        block=_rounding_boundary(primary+correction*(1./2048.))
        acc+=block*(1./256.)
    if HAS_BIAS:
        acc+=tl.load(Bias+cols,cols<N,other=0.)[None,:]
    offset=(batch[:,None]*N+cols[None,:])*S+spatial[:,None]
    tl.store(Y+offset,acc,(rows[:,None]<M)&(cols[None,:]<N))


def pointwise_nchw(value,parts,bias=None,*,block_m=64,block_n=64,
                   block_k=64,group_m=32,num_warps=4,num_stages=2,
                   return_kernel=False,debug_poison=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim!=4 or not value.is_contiguous()):
        raise ValueError('Expected contiguous CUDA FP32 NCHW inference')
    batch,inner,height,width=value.shape
    if len(parts)!=2 or any(p.ndim!=2 or p.shape!=parts[0].shape
            or p.shape[1]!=inner or p.dtype!=torch.float16 or p.device!=value.device
            or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid pointwise compensated weights')
    columns=parts[0].shape[0]
    if min(batch,inner,height,width,columns)<=0:
        raise ValueError('Empty shapes unsupported')
    if bias is not None and (bias.shape!=(columns,) or bias.dtype!=torch.float32
            or bias.device!=value.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    if (block_m not in {32,64} or block_n not in {32,64,128}
            or block_k not in {32,64} or group_m not in {4,8,16,32}
            or num_warps not in {4,8} or num_stages not in {2,3}):
        raise ValueError('Unsupported pointwise configuration')
    rows=batch*height*width
    result=torch.empty((batch,columns,height,width),device=value.device,dtype=torch.float32)
    if debug_poison:result.fill_(float('nan'))
    kernel=_pointwise_nchw[(triton.cdiv(rows,block_m)*triton.cdiv(columns,block_n),)](
        value,*parts,parts[0] if bias is None else bias,result,rows,columns,inner,
        height*width,bias is not None,block_m,block_n,block_k,group_m,
        num_warps=num_warps,num_stages=num_stages,enable_fp_fusion=False)
    return (result,kernel) if return_kernel else result
