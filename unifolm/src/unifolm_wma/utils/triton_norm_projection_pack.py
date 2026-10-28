"""Diagnostic only: original GN statistics, affine/layout/Half split fusion."""
import torch
import triton
import triton.language as tl
from unifolm_wma.utils.triton_split_group_norm import _partial, _finish_stats
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary
from unifolm_wma.utils.triton_half_compensated_linear import _half_split
from unifolm_wma.utils.triton_presplit_two_product_linear import _presplit_two_product_gemm


@triton.jit
def _apply_layout_pack(X, Weight, Bias, Moments, PH, PL, Debug,
                       M:tl.constexpr, C:tl.constexpr, G:tl.constexpr,
                       T:tl.constexpr, H:tl.constexpr, W:tl.constexpr,
                       S0:tl.constexpr, S1:tl.constexpr, ST:tl.constexpr,
                       SH:tl.constexpr, SW:tl.constexpr, TEMPORAL:tl.constexpr,
                       DEBUG:tl.constexpr, RM:tl.constexpr, RC:tl.constexpr):
    row=tl.program_id(0)*RM+tl.arange(0,RM)
    channel=tl.program_id(1)*RC+tl.arange(0,RC)
    if TEMPORAL:
        frame=row%T
        spatial=(row//T)%(H*W)
        batch=row//(T*H*W)
    else:
        frame=tl.full((RM,),0,tl.int32)
        spatial=row%(H*W)
        batch=row//(H*W)
    pos=(batch[:,None]*S0+channel[None,:]*S1+frame[:,None]*ST+
         (spatial[:,None]//W)*SH+(spatial[:,None]%W)*SW)
    valid=(row[:,None]<M)&(channel[None,:]<C)
    value=tl.load(X+pos,valid,other=0).to(tl.float32)
    moment=batch[:,None]*G+channel[None,:]//(C//G)
    mean=tl.load(Moments+moment*2,valid,other=0).to(tl.float32)
    inverse=tl.load(Moments+moment*2+1,valid,other=0).to(tl.float32)
    weight=tl.load(Weight+channel,channel<C,other=0).to(tl.float32)
    bias=tl.load(Bias+channel,channel<C,other=0).to(tl.float32)
    scale=inverse*weight[None,:]
    shift=tl.math.fma(-scale,mean,bias[None,:])
    normalized=tl.math.fma(scale,value,shift)
    normalized=_rounding_boundary(normalized)
    high,low=_half_split(normalized,1.)
    dest=row[:,None]*C+channel[None,:]
    tl.store(PH+dest,high,valid)
    tl.store(PL+dest,low,valid)
    if DEBUG:tl.store(Debug+dest,normalized,valid)


def pack(value,norm,*,tile=(32,32),out=None,debug=None):
    assert not torch.is_grad_enabled() and not norm.training
    assert value.is_cuda and value.dtype==torch.float32 and value.ndim in (4,5)
    b,c=value.shape[:2];g=norm.num_groups;temporal=value.ndim==5
    if temporal:t,h,w=value.shape[2:];s0,s1,st,sh,sw=value.stride()
    else:t,h,w=1,*value.shape[2:];s0,s1,sh,sw=value.stride();st=0
    m=b*t*h*w;size=c//g*t*h*w;parts=triton.cdiv(size,4096)
    stats=torch.empty((b*g,parts,2),device=value.device,dtype=torch.float32)
    moments=torch.empty((b*g,2),device=value.device,dtype=torch.float32)
    if out is None:out=tuple(torch.empty((m,c),device=value.device,dtype=torch.float16) for _ in range(2))
    assert all(x.shape==(m,c) and x.is_contiguous() and x.dtype==torch.float16 for x in out)
    geometry=(g,c,t,h,w,s0,s1,st,sh,sw)
    _partial[(b*g,parts)](value,stats,size,parts,*geometry,4096,num_warps=4,enable_fp_fusion=False)
    _finish_stats[(b*g,)](stats,moments,size,parts,norm.eps,False,4096,triton.next_power_of_2(parts),num_warps=4,enable_fp_fusion=False)
    rm,rc=tile
    kernel=_apply_layout_pack[(triton.cdiv(m,rm),triton.cdiv(c,rc))](value,norm.weight,norm.bias,moments,*out,out[0] if debug is None else debug,m,c,g,t,h,w,s0,s1,st,sh,sw,temporal,debug is not None,rm,rc,num_warps=4,enable_fp_fusion=False)
    return out,kernel


def fused(value,norm,parts,bias,*,tile=(32,32),config=(64,64,32,4,3),output=None,input_parts=None,debug=None,inspect=False):
    packed,packing=pack(value,norm,tile=tile,out=input_parts,debug=debug)
    m,k=packed[0].shape;n=parts[0].shape[0]
    if output is None:output=torch.empty((m,n),device=value.device,dtype=torch.float32)
    bm,bn,group,warps,stages=config
    kernel=_presplit_two_product_gemm[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](*packed,*parts,parts[0] if bias is None else bias,output,m,n,k,bias is not None,True,bm,bn,64,group,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return (output,packed,packing,kernel) if inspect else output


def fused_with_mc(value,norm,parts,bias,*,tile=(32,32),config=(64,64,32,4,3),output=None,input_parts=None,debug=None,inspect=False):
    # A valid FP32 MC tensor keeps the existing projection's fallback semantics.
    if debug is None:
        c=value.shape[1]
        debug=torch.empty((value.numel()//c,c),device=value.device,dtype=torch.float32)
    return fused(value,norm,parts,bias,tile=tile,config=config,output=output,input_parts=input_parts,debug=debug,inspect=inspect)
