"""Keep original per4096 reduction trees; merge small groups into one CTA."""
import torch
import triton
import triton.language as tl
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary
from unifolm_wma.utils import norm_buffer_allocation


@triton.jit
def _serial_norm(X, Weight, Bias, Output, Stats, Moments,
                 SIZE:tl.constexpr, PARTS:tl.constexpr, PART_BLOCK:tl.constexpr,
                 GROUPS:tl.constexpr, CHANNELS:tl.constexpr,
                 FRAMES:tl.constexpr, HEIGHT:tl.constexpr, WIDTH:tl.constexpr,
                 S0:tl.constexpr, S1:tl.constexpr, ST:tl.constexpr,
                 SH:tl.constexpr, SW:tl.constexpr, EPS:tl.constexpr,
                 SILU:tl.constexpr, DEBUG:tl.constexpr):
    row=tl.program_id(0)
    inner=tl.arange(0,4096)
    part=tl.arange(0,PART_BLOCK)
    means=tl.full((PART_BLOCK,),0.,tl.float32)
    squares=tl.full((PART_BLOCK,),0.,tl.float32)
    spatial=FRAMES*HEIGHT*WIDTH
    for index in range(PARTS):
        off=index*4096+inner
        valid=off<SIZE
        channel=(row%GROUPS)*(CHANNELS//GROUPS)+off//spatial
        rem=off%spatial
        position=((row//GROUPS)*S0+channel*S1+(rem//(HEIGHT*WIDTH))*ST+
                  ((rem//WIDTH)%HEIGHT)*SH+(rem%WIDTH)*SW)
        value=tl.load(X+position,valid,other=0).to(tl.float32)
        count=tl.minimum(4096,SIZE-index*4096)
        mean=tl.sum(value,0)/count
        centered=tl.where(valid,value-mean,0.)
        m2=tl.sum(centered*centered,0)
        mean=_rounding_boundary(mean);m2=_rounding_boundary(m2)
        means=tl.where(part==index,mean,means)
        squares=tl.where(part==index,m2,squares)
    counts=tl.where(part<PARTS,tl.minimum(4096,SIZE-part*4096),0)
    overall=tl.sum(means*counts,0)/SIZE
    delta=means-overall
    variance=tl.sum(squares+delta*delta*counts,0)/SIZE
    inverse=tl.math.rsqrt(variance+tl.full((),EPS,tl.float32))
    overall=_rounding_boundary(overall);inverse=_rounding_boundary(inverse)
    # Independent 1024-vector output phase keeps dense vector stores separate
    # from the original 4096-vector partial reduction layout.
    element=tl.arange(0,1024)
    for chunk in range(tl.cdiv(SIZE,1024)):
        off=chunk*1024+element
        valid=off<SIZE
        channel=(row%GROUPS)*(CHANNELS//GROUPS)+off//spatial
        rem=off%spatial
        position=((row//GROUPS)*S0+channel*S1+(rem//(HEIGHT*WIDTH))*ST+
                  ((rem//WIDTH)%HEIGHT)*SH+(rem%WIDTH)*SW)
        value=tl.load(X+position,valid,other=0).to(tl.float32)
        weight=tl.load(Weight+channel,valid,other=0).to(tl.float32)
        bias=tl.load(Bias+channel,valid,other=0).to(tl.float32)
        scale=inverse*weight
        shift=tl.math.fma(-scale,overall,bias)
        normalized=tl.math.fma(scale,value,shift)
        result=normalized
        if SILU:result=normalized/(1.+tl.exp(-normalized))
        tl.store(Output+row*SIZE+off,result,valid)
    if DEBUG:
        tl.store(Stats+(row*PARTS+part)*2,means,part<PARTS)
        tl.store(Stats+(row*PARTS+part)*2+1,squares,part<PARTS)
        tl.store(Moments+row*2,overall);tl.store(Moments+row*2+1,inverse)


def fused(value,norm,*,apply_silu=True,debug=False):
    assert not torch.is_grad_enabled() and not norm.training
    assert value.is_cuda and value.dtype==torch.float32 and value.ndim in (4,5)
    batch,channels=value.shape[:2];groups=norm.num_groups
    size=value.numel()//(batch*groups);parts=triton.cdiv(size,4096)
    assert 0<size<=32768 and channels==norm.num_channels
    assert all(p is not None and p.dtype==value.dtype and p.device==value.device and p.is_contiguous() for p in (norm.weight,norm.bias))
    if value.ndim==4:
        frames,height,width=1,*value.shape[2:];s0,s1,sh,sw=value.stride();st=0
    else:frames,height,width=value.shape[2:];s0,s1,st,sh,sw=value.stride()
    output=norm_buffer_allocation.empty('group',value.shape,device=value.device,dtype=value.dtype)
    stats=norm_buffer_allocation.empty('group',(batch*groups,parts,2),device=value.device,dtype=value.dtype) if debug else output
    moments=norm_buffer_allocation.empty('group',(batch*groups,2),device=value.device,dtype=value.dtype) if debug else output
    compiled=_serial_norm[(batch*groups,)](value,norm.weight,norm.bias,output,stats,moments,
        size,parts,triton.next_power_of_2(parts),groups,channels,frames,height,width,s0,s1,st,sh,sw,norm.eps,apply_silu,debug,
        num_warps=4,enable_fp_fusion=False)
    return (output,stats,moments,compiled) if debug else output
