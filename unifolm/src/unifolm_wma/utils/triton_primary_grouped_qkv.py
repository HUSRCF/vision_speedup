"""Diagnostic primary C320 QKV GEMM with three contiguous output planes."""
import torch
from .gemm_output_allocation import empty as empty_gemm_output
import triton
import triton.language as tl


@triton.jit
def _grouped_qkv(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
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
    offset=(cols[None,:]//N)*M*N+rows[:,None]*N+cols[None,:]%N
    tl.store(Y+offset,acc,(rows[:,None]<M)&(cols[None,:]<3*N))


def primary_grouped_qkv(value,high_weight,*,tile=(128,64,32,4,2),debug_poison=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim<1 or not value.is_contiguous() or value.shape[-1]==0
            or high_weight.ndim!=2 or high_weight.dtype!=torch.float16
            or high_weight.device!=value.device or not high_weight.is_contiguous()
            or high_weight.shape[0]%3 or high_weight.shape[1]!=value.shape[-1]):
        raise ValueError('Expected FP32 CUDA input and three stacked scaled Half weights')
    rows,inner=value.numel()//value.shape[-1],value.shape[-1]
    width=high_weight.shape[0]//3
    if min(rows,inner,width)<=0:raise ValueError('Empty QKV dimensions')
    bm,bn,group,warps,stages=tile
    if (bm not in {64,128} or bn not in {64,128} or group not in {16,32}
            or warps not in {4,8} or stages not in {2,3}):
        raise ValueError('Unsupported grouped QKV tile')
    output=empty_gemm_output('primary',(3,*value.shape[:-1],width),device=value.device,dtype=torch.float32)
    if debug_poison:output.fill_(float('nan'))
    _grouped_qkv[(triton.cdiv(rows,bm)*triton.cdiv(3*width,bn),)](
        value,high_weight,output,rows,width,inner,bm,bn,group,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return output
