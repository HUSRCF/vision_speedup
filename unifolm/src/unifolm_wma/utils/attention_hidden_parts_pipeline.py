import torch
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary
from .triton_compensated_attention import _three_product
from .half_pack_allocation import empty_like as empty_half
from .gemm_output_allocation import empty as empty_gemm

@triton.jit
def _attention_hidden_parts_vector(Q, KH, KL, VH, VL, OH, OL,
                        M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
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
        offsets=batch*N*H*64+position[None,:]*H*64+head*64+channel[:,None]
        k_high=tl.load(KH+offsets,position[None,:]<N,other=0)
        k_low=tl.load(KL+offsets,position[None,:]<N,other=0)
        scores=_three_product(q_high,q_low,k_high,k_low,1.0)*SCALE
        scores=tl.where(position[None,:]<N,scores,-float('inf'))
        next_maximum=tl.maximum(maximum,tl.max(scores,1))
        alpha=tl.exp(maximum-next_maximum)
        probabilities=tl.exp(scores-next_maximum[:,None])
        next_denominator=denominator*alpha+tl.sum(probabilities,1)
        accumulator=accumulator*alpha[:,None]
        p_high,p_low=_half_split(probabilities,256.0)
        offsets=batch*N*H*64+position[:,None]*H*64+head*64+channel[None,:]
        v_high=tl.load(VH+offsets,position[:,None]<N,other=0)
        v_low=tl.load(VL+offsets,position[:,None]<N,other=0)
        accumulator+=_three_product(p_high,p_low,v_high,v_low,1/256.0)
        maximum=next_maximum
        denominator=next_denominator
    output=accumulator/denominator[:,None]
    high,low=_half_split(_rounding_boundary(output),1.)
    offset=batch*M*H*64+query[:,None]*H*64+head*64+channel[None,:]
    tl.store(OH+offset,high,query[:,None]<M)
    tl.store(OL+offset,low,query[:,None]<M)


@triton.jit
def _attention_hidden_parts_barrier(Q, KH, KL, VH, VL, OH, OL,
                        M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
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
        offsets=batch*N*H*64+position[None,:]*H*64+head*64+channel[:,None]
        k_high=tl.load(KH+offsets,position[None,:]<N,other=0)
        k_low=tl.load(KL+offsets,position[None,:]<N,other=0)
        scores=_three_product(q_high,q_low,k_high,k_low,1.0)*SCALE
        scores=tl.where(position[None,:]<N,scores,-float('inf'))
        next_maximum=tl.maximum(maximum,tl.max(scores,1))
        alpha=tl.exp(maximum-next_maximum)
        probabilities=tl.exp(scores-next_maximum[:,None])
        next_denominator=denominator*alpha+tl.sum(probabilities,1)
        accumulator=accumulator*alpha[:,None]
        p_high,p_low=_half_split(probabilities,256.0)
        offsets=batch*N*H*64+position[:,None]*H*64+head*64+channel[None,:]
        v_high=tl.load(VH+offsets,position[:,None]<N,other=0)
        v_low=tl.load(VL+offsets,position[:,None]<N,other=0)
        accumulator+=_three_product(p_high,p_low,v_high,v_low,1/256.0)
        maximum=next_maximum
        denominator=next_denominator
    output=accumulator/denominator[:,None]
    high,low=_half_split(_rounding_boundary(output),1.)
    offset=batch*M*H*64+query[:,None]*H*64+head*64+channel[None,:]
    offset=tl.inline_asm_elementwise("mov.b32 $0, $1;",constraints="=r,r",
        args=[offset],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(OH+offset,high,query[:,None]<M)
    tl.store(OL+offset,low,query[:,None]<M)


@triton.jit
def _packed_attention_output_gemm(XH, XL, High, Low, Bias, Output,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      HAS_BIAS: tl.constexpr, INPUT_PROTECTED: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles = tl.cdiv(M, BM)
    col_tiles = tl.cdiv(N, BN)
    group = pid // (GROUP_M * col_tiles)
    first_row = group * GROUP_M
    group_rows = tl.minimum(GROUP_M, row_tiles - first_row)
    within_group = pid % (GROUP_M * col_tiles)
    rows = (first_row + within_group % group_rows) * BM + tl.arange(0, BM)
    cols = (within_group // group_rows) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        x_high=tl.load(XH+rows[:,None]*K+k[None,:],
                       (rows[:,None]<M)&(k[None,:]<K),other=0)
        if INPUT_PROTECTED:
            x_low=tl.load(XL+rows[:,None]*K+k[None,:],
                          (rows[:,None]<M)&(k[None,:]<K),other=0)
        w_high = tl.load(High + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
        primary = _rounding_boundary(tl.dot(x_high, w_high))
        if INPUT_PROTECTED:
            correction = tl.dot(x_low, w_high)
        else:
            w_low = tl.load(Low + cols[None, :] * K + k[:, None],
                            (cols[None, :] < N) & (k[:, None] < K), other=0)
            correction = tl.dot(x_high, w_low)
        block = _rounding_boundary(primary + correction * (1 / 2048.0))
        accumulator += block * (1 / 256.0)
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    tl.store(Output + rows[:, None] * N + cols[None, :], accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))


def attention_hidden_parts(query,kv,*,mode='vector'):
    batch,rows,heads,width=query.shape
    assert query.is_cuda and query.dtype==torch.float32 and query.is_contiguous() and width==64
    assert mode in {'vector','barrier'} and len(kv)==4
    length=kv[0].numel()//(batch*heads*64)
    assert all(p.dtype==torch.float16 and p.device==query.device and p.numel()==batch*length*heads*64 and p.is_contiguous() for p in kv)
    outputs=tuple(empty_half(query) for _ in range(2))
    kernel={'vector':_attention_hidden_parts_vector,'barrier':_attention_hidden_parts_barrier}[mode]
    kernel[(triton.cdiv(rows,64),batch*heads)](query,*kv,*outputs,rows,length,heads,.125,64,64,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return outputs


def packed_attention_output(inputs,parts,bias=None):
    high,low=inputs;batch,length,heads,width=high.shape
    rows=batch*length;inner=heads*width;channels=parts[0].shape[0]
    assert high.dtype==low.dtype==torch.float16 and high.shape==low.shape and width==64
    assert all(p.shape==(channels,inner) and p.dtype==torch.float16 and p.device==high.device and p.is_contiguous() for p in parts)
    import os
    mode=os.environ.get('UNIFOLM_WORLD_ATTENTION_PROJECTION_SCHEDULE','0')
    if mode not in {'0','1'}:
        raise ValueError('Invalid attention projection schedule')
    selected=(mode=='1' and not torch.is_grad_enabled()
        and (batch,length,heads,width,channels) in {(16,2560,5,64,320),(16,640,10,64,640)}
        and bias is not None and bias.dtype==torch.float32 and bias.device==high.device and bias.is_contiguous()
        and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
    group,stages=(8,3) if selected else (32,2)
    output=empty_gemm('classic',(batch,length,channels),device=high.device,dtype=torch.float32)
    _packed_attention_output_gemm[(triton.cdiv(rows,64)*triton.cdiv(channels,64),)](
        high,low,*parts,parts[0] if bias is None else bias,output,rows,channels,inner,bias is not None,True,
        64,64,64,group,num_warps=4,num_stages=stages,enable_fp_fusion=False)
    return output
