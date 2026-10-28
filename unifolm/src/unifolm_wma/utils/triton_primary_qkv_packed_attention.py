"""Decision primary QKV and K/V packing, preserving the accepted dot chain."""
import torch
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split
from .triton_prepacked_compensated_attention import _prepacked_attention
from .gemm_output_allocation import empty as empty_gemm
from .half_pack_allocation import empty_like as empty_half
from .attention_buffer_allocation import empty_like as empty_attention


@triton.jit
def _primary_qkv_pack(X,W,Q,KH,KL,VH,VL,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                      BM:tl.constexpr,BN:tl.constexpr,GROUP:tl.constexpr):
    pid=tl.program_id(0)
    mt,nt=tl.cdiv(M,BM),tl.cdiv(3*N,BN)
    group=pid//(GROUP*nt)
    first=group*GROUP
    count=tl.minimum(GROUP,mt-first)
    local=pid%(GROUP*nt)
    rows=(first+local%count)*BM+tl.arange(0,BM)
    cols=(local//count)*BN+tl.arange(0,BN)
    inner=tl.arange(0,64)
    acc=tl.full((BM,BN),0.,tl.float32)
    for block in range(tl.cdiv(K,64)):
        k=block*64+inner
        x=tl.load(X+rows[:,None]*K+k[None,:],
            (rows[:,None]<M)&(k[None,:]<K),other=0.).to(tl.float16)
        w=tl.load(W+cols[None,:]*K+k[:,None],
            (cols[None,:]<3*N)&(k[:,None]<K),other=0.)
        acc=tl.dot(x,w,acc)
    acc*=1./256.
    high,low=_half_split(acc,1.)
    offset=rows[:,None]*N+cols[None,:]%N
    valid=(rows[:,None]<M)&(cols[None,:]<3*N)
    tl.store(Q+offset,acc,valid&(cols[None,:]<N))
    tl.store(KH+offset,high,valid&(cols[None,:]>=N)&(cols[None,:]<2*N))
    tl.store(KL+offset,low,valid&(cols[None,:]>=N)&(cols[None,:]<2*N))
    tl.store(VH+offset,high,valid&(cols[None,:]>=2*N))
    tl.store(VL+offset,low,valid&(cols[None,:]>=2*N))


def primary_qkv_packed_attention(value,high_weight,*,scale=.125,debug=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim!=3 or not value.is_contiguous() or value.shape[-1]!=320
            or min(value.shape)<=0 or high_weight.shape!=(960,320)
            or high_weight.dtype!=torch.float16 or high_weight.device!=value.device
            or not high_weight.is_contiguous() or scale!=.125):
        raise ValueError('Expected accepted C320 primary QKV and H64 attention inputs')
    batch,length,channels=value.shape
    query=empty_gemm('primary',value.shape,device=value.device,dtype=torch.float32)
    kh,kl,vh,vl=(empty_half(query) for _ in range(4))
    _primary_qkv_pack[(triton.cdiv(batch*length,128)*triton.cdiv(3*channels,128),)](
        value,high_weight,query,kh,kl,vh,vl,batch*length,channels,channels,128,128,32,
        num_warps=4,num_stages=3,enable_fp_fusion=False)
    heads=channels//64
    shaped=query.view(batch,length,heads,64)
    output=empty_attention('spatial',shaped)
    _prepacked_attention[(triton.cdiv(length,64),batch*heads)](
        shaped,kh,kl,vh,vl,output,length,length,heads,scale,64,64,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    result=output.view(batch,length,channels)
    return (result,(query,kh,kl,vh,vl)) if debug else result
