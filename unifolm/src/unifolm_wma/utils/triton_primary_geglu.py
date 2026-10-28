"""Diagnostic lower-precision projection fused with accepted FP32 GEGLU.

Mode value protects the gate's accepted three-product short-K arithmetic.
Mode both uses one Half product per branch. Neither mode changes inference
unless a separate guarded runtime installer is added and quality is verified.
"""
import torch
from .geglu_output_allocation import empty as empty_geglu_output
import triton
import triton.language as tl
from .triton_half_compensated_linear import pack_half_weight
from .triton_prepacked_geglu import _short_product


@triton.jit
def _primary_geglu(XH,XL,WH,WL,Bias,Y,M:tl.constexpr,C:tl.constexpr,
                   K:tl.constexpr,HAS_BIAS:tl.constexpr,BM:tl.constexpr,
                   BN:tl.constexpr,GROUP_M:tl.constexpr,PRIMARY_GATE:tl.constexpr):
    pid=tl.program_id(0)
    row_tiles,col_tiles=tl.cdiv(M,BM),tl.cdiv(C,BN)
    group=pid//(GROUP_M*col_tiles)
    first_row=group*GROUP_M
    group_rows=tl.minimum(GROUP_M,row_tiles-first_row)
    local=pid%(GROUP_M*col_tiles)
    rows=(first_row+local%group_rows)*BM+tl.arange(0,BM)
    cols=(local//group_rows)*BN+tl.arange(0,BN)
    inner=tl.arange(0,64)
    value=tl.full((BM,BN),0.,tl.float32)
    gate=tl.full((BM,BN),0.,tl.float32)
    for start in range(tl.cdiv(K,64)):
        k=start*64+inner
        xoffset=rows[:,None]*K+k[None,:]
        xvalid=(rows[:,None]<M)&(k[None,:]<K)
        xh=tl.load(XH+xoffset,xvalid,other=0.)
        offset=cols[None,:]*K+k[:,None]
        valid=(cols[None,:]<C)&(k[:,None]<K)
        vh=tl.load(WH+offset,valid,other=0.)
        gh=tl.load(WH+offset+C*K,valid,other=0.)
        value=tl.dot(xh,vh,value)
        if PRIMARY_GATE:
            gate=tl.dot(xh,gh,gate)
        else:
            xl=tl.load(XL+xoffset,xvalid,other=0.)
            gl=tl.load(WL+offset+C*K,valid,other=0.)
            gate+=_short_product(xh,xl,gh,gl)
    value=value*(1./256.)
    if PRIMARY_GATE:gate=gate*(1./256.)
    if HAS_BIAS:
        value+=tl.load(Bias+cols,cols<C,other=0.)[None,:]
        gate+=tl.load(Bias+cols+C,cols<C,other=0.)[None,:]
    gelu=(gate*.5)*(1.+tl.math.erf(gate*.7071067811865476))
    tl.store(Y+rows[:,None]*C+cols[None,:],value*gelu,
             (rows[:,None]<M)&(cols[None,:]<C))


def primary_geglu(value,parts,bias=None,*,mode='value',debug_poison=False):
    if mode not in {'value','both'}:raise ValueError('Unknown primary GEGLU mode')
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim<1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous CUDA FP32 inference input')
    if len(parts)!=2 or any(p.ndim!=2 or p.shape!=parts[0].shape
            or p.shape[1]!=value.shape[-1] or p.shape[0]%2 or p.dtype!=torch.float16
            or p.device!=value.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid GEGLU compensated weights')
    channels=parts[0].shape[0]//2
    inner=value.shape[-1]
    rows=value.numel()//inner
    if min(rows,channels,inner)<=0:raise ValueError('Empty shapes unsupported')
    if bias is not None and (bias.shape!=(2*channels,) or bias.dtype!=torch.float32
            or bias.device!=value.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    xh,xl=pack_half_weight(value.view(rows,inner),weight_scale=1.)
    output=empty_geglu_output((*value.shape[:-1],channels),device=value.device,dtype=torch.float32)
    if debug_poison:output.fill_(float('nan'))
    bn=64 if inner==640 else 32
    _primary_geglu[(triton.cdiv(rows,64)*triton.cdiv(channels,bn),)](
        xh,xl,*parts,parts[0] if bias is None else bias,output,rows,channels,inner,
        bias is not None,64,bn,32,mode=='both',num_warps=4,num_stages=2,
        enable_fp_fusion=False)
    return output
