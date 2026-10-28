"""Experimental Half projection/GEGLU fusion; original master tensors are read."""
import torch
import triton
import triton.language as tl


@triton.jit
def _project_activate(X, W, Bias, Y, M:tl.constexpr, C:tl.constexpr, H:tl.constexpr,
                      BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM);nn=tl.cdiv(H,BN);group_size=8*nn
    group=pid//group_size;first=group*8;actual=tl.minimum(nm-first,8)
    mi=first+(pid%group_size)%actual;ni=(pid%group_size)//actual
    rows=mi*BM+tl.arange(0,BM)
    cols=ni*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    av=tl.full((BM,BN),0,tl.float32);ag=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(C,BK)):
        k=start*BK+kk
        x=tl.load(X+rows[:,None]*C+k[None,:],(rows[:,None]<M)&(k[None,:]<C),other=0)
        wv=tl.load(W+cols[None,:]*C+k[:,None],(cols[None,:]<H)&(k[:,None]<C),other=0)
        wg=tl.load(W+(cols[None,:]+H)*C+k[:,None],(cols[None,:]<H)&(k[:,None]<C),other=0)
        av=tl.dot(x,wv,av);ag=tl.dot(x,wg,ag)
    bv=tl.load(Bias+cols,cols<H,other=0).to(tl.float32)
    bg=tl.load(Bias+cols+H,cols<H,other=0).to(tl.float32)
    # F.linear's materialized Half projection precedes the existing exact-GELU
    # activation. Preserve those two rounding boundaries and the final cast.
    value=(av+bv[None,:]).to(tl.float16).to(tl.float32)
    gate=(ag+bg[None,:]).to(tl.float16).to(tl.float32)
    gelu=((gate*.5)*(1.+tl.math.erf(gate*.7071067811865476))).to(tl.float16).to(tl.float32)
    tl.store(Y+rows[:,None]*H+cols[None,:],value*gelu,(rows[:,None]<M)&(cols[None,:]<H))


def fused(value,weight,bias,config=(64,64,32,4,3)):
    assert not torch.is_grad_enabled() and not torch.is_autocast_enabled()
    assert value.is_cuda and value.dtype==weight.dtype==bias.dtype==torch.float16
    assert value.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
    assert value.device==weight.device==bias.device
    c=value.shape[-1];h=weight.shape[0]//2;m=value.numel()//c
    assert tuple(weight.shape)==(2*h,c) and tuple(bias.shape)==(2*h,)
    bm,bn,bk,warps,stages=config
    output=torch.empty((*value.shape[:-1],h),dtype=value.dtype,device=value.device)
    _project_activate[(triton.cdiv(m,bm)*triton.cdiv(h,bn),)](
        value,weight,bias,output,m,c,h,bm,bn,bk,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return output
