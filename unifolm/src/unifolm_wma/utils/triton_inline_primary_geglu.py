"""One-product GEGLU with inline input casting and FP32 bias/activation/output."""
import torch
import triton
import triton.language as tl
from .triton_native_geglu_activation import native_geglu


@triton.jit
def _inline_primary_geglu(X, WH, Bias, Y, V, G,
                          M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                          HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                          GROUP: tl.constexpr, DEBUG: tl.constexpr,
                          NATIVE_ACTIVATION: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(C, BN)
    group = pid // (GROUP * col_tiles)
    first = group * GROUP
    count = tl.minimum(GROUP, row_tiles-first)
    local = pid % (GROUP * col_tiles)
    rows = (first+local%count)*BM+tl.arange(0,BM)
    cols = (local//count)*BN+tl.arange(0,BN)
    inner = tl.arange(0,64)
    value = tl.full((BM,BN),0.,tl.float32)
    gate = tl.full((BM,BN),0.,tl.float32)
    for block in range(tl.cdiv(K,64)):
        k = block*64+inner
        x = tl.load(X+rows[:,None]*K+k[None,:],
            (rows[:,None]<M)&(k[None,:]<K),other=0).to(tl.float16)
        offset = cols[None,:]*K+k[:,None]
        valid = (cols[None,:]<C)&(k[:,None]<K)
        wv = tl.load(WH+offset,valid,other=0)
        wg = tl.load(WH+offset+C*K,valid,other=0)
        value = tl.dot(x,wv,value)
        gate = tl.dot(x,wg,gate)
    value *= 1./256.
    gate *= 1./256.
    if HAS_BIAS:
        value += tl.load(Bias+cols,cols<C,other=0)[None,:]
        gate += tl.load(Bias+cols+C,cols<C,other=0)[None,:]
    offset = rows[:,None]*C+cols[None,:]
    valid = (rows[:,None]<M)&(cols[None,:]<C)
    if DEBUG:
        tl.store(V+offset,value,valid)
        tl.store(G+offset,gate,valid)
    if NATIVE_ACTIVATION:
        result = native_geglu(value,gate)
    else:
        activated = (gate*.5)*(1.+tl.math.erf(gate*.7071067811865476))
        result = value*activated
    tl.store(Y+offset,result,valid)


def inline_primary_geglu(value, high_weight, bias=None, *,
                          tile=(64,64,32,4,2), debug_branches=False, debug_poison=False,
                          activation='native'):
    if activation not in {'native','triton'}:
        raise ValueError('Unsupported GEGLU activation')
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim<1 or not value.is_contiguous()
            or high_weight.ndim!=2 or high_weight.dtype!=torch.float16
            or high_weight.device!=value.device or not high_weight.is_contiguous()
            or high_weight.shape[1]!=value.shape[-1] or high_weight.shape[0]%2):
        raise ValueError('Expected FP32 CUDA input and contiguous scaled Half GEGLU weights')
    rows, inner = value.numel()//value.shape[-1],value.shape[-1]
    channels = high_weight.shape[0]//2
    if min(rows,inner,channels)<=0:
        raise ValueError('Empty GEGLU dimensions unsupported')
    if bias is not None and (bias.shape!=(2*channels,) or bias.dtype!=torch.float32
            or bias.device!=value.device or not bias.is_contiguous()):
        raise ValueError('Invalid GEGLU FP32 bias')
    bm,bn,group,warps,stages = tile
    if (bm not in {32,64,128} or bn not in {32,64,128} or group not in {16,32}
            or warps not in {4,8} or stages not in {2,3}):
        raise ValueError('Unsupported inline GEGLU tile')
    shape = (*value.shape[:-1],channels)
    output = torch.empty(shape,device=value.device,dtype=torch.float32)
    v = torch.empty_like(output) if debug_branches else output
    g = torch.empty_like(output) if debug_branches else output
    if debug_poison:
        for tensor in (output,v,g):
            tensor.fill_(float('nan'))
    _inline_primary_geglu[(triton.cdiv(rows,bm)*triton.cdiv(channels,bn),)](
        value,high_weight,high_weight if bias is None else bias,output,v,g,
        rows,channels,inner,bias is not None,bm,bn,group,debug_branches,activation=='native',
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return (output,v,g) if debug_branches else output
