"""Fuse accepted QKV short sums with the existing K/V Half decomposition.

The attention kernel and its reduction order remain unchanged. The K/V FP32
intermediates are replaced by the exact Half pairs they would produce next.
"""
import torch
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split,pack_half_weight
from .triton_zero_weight_low_linear import _short_sum
from .triton_prepacked_compensated_attention import _prepacked_attention
from .gemm_output_allocation import empty as empty_gemm
from .half_pack_allocation import empty_like as empty_half
from .attention_buffer_allocation import empty_like as empty_attention


@triton.jit
def _qkv_pack(XH,XL,WH,WL,Count,Q,KH,KL,VH,VL,
              M:tl.constexpr,C:tl.constexpr,BM:tl.constexpr):
    pid=tl.program_id(0)
    mt,nt=tl.cdiv(M,BM),tl.cdiv(C*3,64)
    group=pid//(32*nt);first=group*32
    size=tl.minimum(32,mt-first);local=pid%(32*nt)
    rows=(first+local%size)*BM+tl.arange(0,BM)
    tile=local//size
    cols=tile*64+tl.arange(0,64)
    if tl.load(Count)!=0:
        value=_short_sum(XH,XL,WH,WL,rows,cols,M,C*3,C,BM,True)
    else:
        value=_short_sum(XH,XL,WH,WL,rows,cols,M,C*3,C,BM,False)
    component=tile//(C//64)
    offset=rows[:,None]*C+cols[None,:]%C
    mask=rows[:,None]<M
    if component==0:
        tl.store(Q+offset,value,mask)
    else:
        high,low=_half_split(value,1.)
        if component==1:
            tl.store(KH+offset,high,mask);tl.store(KL+offset,low,mask)
        else:
            tl.store(VH+offset,high,mask);tl.store(VL+offset,low,mask)


def qkv_packed_attention(value,parts,count,*,scale=.125,debug=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim!=3 or not value.is_contiguous() or value.shape[-1] not in {320,640}
            or min(value.shape)<=0):
        raise ValueError('Expected contiguous CUDA FP32 BMC inference C320/640')
    batch,length,channels=value.shape
    if len(parts)!=2 or any(p.shape!=(channels*3,channels) or p.dtype!=torch.float16
            or p.device!=value.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid accepted packed QKV master parts')
    if count.shape!=() or count.dtype!=torch.int64 or count.device!=value.device:
        raise ValueError('Invalid graph-visible weight residual predicate')
    if scale!=.125:raise ValueError('Expected accepted H64 spatial attention scale')
    rows=batch*length
    packed=pack_half_weight(value.view(rows,channels),weight_scale=1.)
    query=empty_gemm('classic',value.shape,device=value.device,dtype=torch.float32)
    kh,kl,vh,vl=(empty_half(query) for _ in range(4))
    bm=64 if rows>=2560 else 32
    _qkv_pack[(triton.cdiv(rows,bm)*triton.cdiv(channels*3,64),)](
        *packed,*parts,count,query,kh,kl,vh,vl,rows,channels,bm,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    heads=channels//64
    shaped=query.view(batch,length,heads,64)
    output=empty_attention('spatial',shaped)
    _prepacked_attention[(triton.cdiv(length,64),batch*heads)](
        shaped,kh,kl,vh,vl,output,length,length,heads,scale,64,64,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    result=output.view(batch,length,channels)
    return (result,(query,kh,kl,vh,vl)) if debug else result
