"""Opt-in FP32-output streaming attention using three scaled FP16 products.

Queries, keys, values, logits, softmax state and final output retain FP32.
Only the two matrix products use decomposed FP16 operands. Approximate
arithmetic and narrower FP16 exponent range require real-input/full-video
validation before installation. No dropout, bias, masking or backward path.
"""
import os; import torch; from .attention_buffer_allocation import empty_like as empty_attention_output
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary


@triton.jit
def _three_product(lhs_high,lhs_low,rhs_high,rhs_low,SCALE:tl.constexpr):
    primary=_rounding_boundary(tl.dot(lhs_high,rhs_high))
    correction=tl.dot(lhs_low,rhs_high)
    correction=tl.dot(lhs_high,rhs_low,correction)
    return _rounding_boundary(primary+correction*(1/2048.0))*SCALE


@triton.jit
def _compensated_attention(Q,K,V,O,M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
                           SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    query_block=tl.program_id(0)
    batch_head=tl.program_id(1)
    batch=batch_head//H
    head=batch_head%H
    query=query_block*BM+tl.arange(0,BM)
    channel=tl.arange(0,64)
    key=tl.arange(0,BN)
    q=tl.load(Q+batch*M*H*64+query[:,None]*H*64+head*64+channel[None,:],
              query[:,None]<M,other=0)
    q_high,q_low=_half_split(q,1.0)
    maximum=tl.full((BM,),-float('inf'),tl.float32)
    denominator=tl.full((BM,),0.,tl.float32)
    accumulator=tl.full((BM,64),0.,tl.float32)
    for start in range(tl.cdiv(N,BN)):
        position=start*BN+key
        k=tl.load(K+batch*N*H*64+position[None,:]*H*64+head*64+channel[:,None],
                  position[None,:]<N,other=0)
        k_high,k_low=_half_split(k,1.0)
        scores=_three_product(q_high,q_low,k_high,k_low,1.0)*SCALE
        scores=tl.where(position[None,:]<N,scores,-float('inf'))
        next_maximum=tl.maximum(maximum,tl.max(scores,1))
        alpha=tl.exp(maximum-next_maximum)
        probabilities=tl.exp(scores-next_maximum[:,None])
        next_denominator=denominator*alpha+tl.sum(probabilities,1)
        accumulator=accumulator*alpha[:,None]
        # Online probabilities are in [0,1]. The exact power-of-two scale
        # protects their residual from FP16 underflow without clipping.
        p_high,p_low=_half_split(probabilities,256.0)
        v=tl.load(V+batch*N*H*64+position[:,None]*H*64+head*64+channel[None,:],
                  position[:,None]<N,other=0)
        v_high,v_low=_half_split(v,1.0)
        accumulator+=_three_product(p_high,p_low,v_high,v_low,1/256.0)
        maximum=next_maximum
        denominator=next_denominator
    output=accumulator/denominator[:,None]
    tl.store(O+batch*M*H*64+query[:,None]*H*64+head*64+channel[None,:],
             output,query[:,None]<M)


def compensated_attention(query,key,value,*,scale=None,block_m=32,block_n=64,
                            num_warps=4,return_kernel=False):
    if (query.ndim!=4 or key.ndim!=4 or query.shape[-1]!=64 or
            key.shape[-1]!=64 or key.shape!=value.shape or
            query.shape[0]!=key.shape[0] or query.shape[2]!=key.shape[2] or
            min(query.shape[:3])<=0 or key.shape[1]<=0 or
            any(not x.is_cuda or x.dtype!=torch.float32 or not x.is_contiguous()
                or x.device!=query.device for x in (query,key,value)) or
            torch.is_grad_enabled()):
        raise ValueError('Compensated attention requires contiguous CUDA FP32 BM H64 inference')
    if block_m not in (16,32,64) or block_n not in (32,64,128) or num_warps not in (4,8):
        raise ValueError('Unsupported compensated attention configuration')
    batch,rows,heads,_=query.shape
    mode=os.environ.get('UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV','0')
    if mode not in {'0','kv'}:
        raise ValueError('Invalid spatial attention K/V packing scope')
    if mode=='kv' and rows>=160 and key.shape[1]>=160:
        from .triton_prepacked_compensated_attention import prepacked_compensated_attention
        return prepacked_compensated_attention(query,key,value,scale=scale,
            block_m=block_m,block_n=block_n,num_warps=num_warps,return_kernel=return_kernel)
    output=empty_attention_output('spatial', query)
    kernel=_compensated_attention[(triton.cdiv(rows,block_m),batch*heads)](
        query,key,value,output,rows,key.shape[1],heads,
        0.125 if scale is None else scale,block_m,block_n,
        num_warps=num_warps,num_stages=2,enable_fp_fusion=False)
    return (output,kernel) if return_kernel else output
