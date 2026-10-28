"""Fuse temporal indexing and input splitting into the accepted GEMM math."""
import torch
from unifolm_wma.utils.conv_buffer_allocation import empty as empty_conv_buffer
import triton
import triton.language as tl
from unifolm_wma.utils.triton_half_compensated_linear import _half_split
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary



@triton.jit
def _split_ncthw(X, High, Low, SIZE:tl.constexpr, INTERLEAVED:tl.constexpr, BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(X+off,off<SIZE,other=0)
    high,low=_half_split(value,1.)
    if INTERLEAVED:
        bits=high.to(tl.uint16,bitcast=True).to(tl.uint32)
        bits=bits|(low.to(tl.uint16,bitcast=True).to(tl.uint32)<<16)
        tl.store(High+off,bits,off<SIZE)
    else:
        tl.store(High+off,high,off<SIZE)
        tl.store(Low+off,low,off<SIZE)

@triton.jit
def _presplit_temporal(X, XL, WH, WL, Bias, Output, LowCount,
                       M: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
                       T: tl.constexpr, S: tl.constexpr,
                       HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, THREE: tl.constexpr, INTERLEAVED:tl.constexpr):
    if (tl.load(LowCount)!=0)==THREE:
        pid=tl.program_id(0)
        row_tiles,col_tiles=tl.cdiv(M,BM),tl.cdiv(N,BN)
        group=pid//(GROUP_M*col_tiles)
        first_row=group*GROUP_M
        group_rows=tl.minimum(GROUP_M,row_tiles-first_row)
        local=pid%(GROUP_M*col_tiles)
        rows=(first_row+local%group_rows)*BM+tl.arange(0,BM)
        cols=(local//group_rows)*BN+tl.arange(0,BN)
        inner=tl.arange(0,BK)
        batch=rows//(T*S)
        frame,spatial=(rows//S)%T,rows%S
        acc=tl.full((BM,BN),0.,tl.float32)
        for start in range(tl.cdiv(C*3,BK)):
            k=start*BK+inner
            channel,tap=k//3,k%3
            source_frame=frame[:,None]+tap[None,:]-1
            source=((batch[:,None]*C+channel[None,:])*T+source_frame)*S+spatial[:,None]
            valid=(rows[:,None]<M)&(k[None,:]<C*3)&(source_frame>=0)&(source_frame<T)
            if INTERLEAVED:
                bits=tl.load(X+source,valid,other=0).to(tl.uint32)
                xh=(bits&65535).to(tl.uint16).to(tl.float16,bitcast=True)
                xl=(bits>>16).to(tl.uint16).to(tl.float16,bitcast=True)
            else:
                xh=tl.load(X+source,valid,other=0)
                xl=tl.load(XL+source,valid,other=0)
            offset=cols[None,:]*(C*3)+k[:,None]
            valid_w=(cols[None,:]<N)&(k[:,None]<C*3)
            wh=tl.load(WH+offset,valid_w,other=0)
            if THREE:
                wl=tl.load(WL+offset,valid_w,other=0)
            primary=_rounding_boundary(tl.dot(xh,wh))
            correction=tl.dot(xl,wh)
            if THREE:
                correction=tl.dot(xh,wl,correction)
            block=_rounding_boundary(primary+correction*(1./2048.))
            acc+=block*(1./256.)
        if HAS_BIAS:
            acc+=tl.load(Bias+cols,cols<N,other=0)[None,:]
        output=((rows[:,None]//(T*S))*N+cols[None,:])*(T*S)+rows[:,None]%(T*S)
        tl.store(Output+output,acc,(rows[:,None]<M)&(cols[None,:]<N))


def presplit_temporal_conv(value,parts,low_count,bias=None,*,block_m=64,block_n=64,
                            group_m=32,num_stages=2,debug_poison=False,return_kernels=False,interleaved=False,input_parts=None):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim!=5 or (input_parts is None and not value.is_contiguous())):
        raise ValueError('Expected contiguous inference CUDA FP32 NCTHW')
    import os
    schedule=os.environ.get('UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE','2')
    if schedule not in {'2','3'}:
        raise ValueError('Invalid partitioned temporal schedule')
    batch,channels,frames,height,width=value.shape
    if len(parts)!=2 or any(p.ndim!=2 or p.shape!=parts[0].shape
            or p.shape[1]!=channels*3 or p.dtype!=torch.float16
            or p.device!=value.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid compensated temporal weights')
    outputs=parts[0].shape[0]
    if min(batch,channels,outputs,frames,height,width)<=0:
        raise ValueError('Empty dimensions unsupported')
    if bias is not None and (bias.shape!=(outputs,) or bias.dtype!=torch.float32
            or bias.device!=value.device or not bias.is_contiguous()):
        raise ValueError('Invalid temporal FP32 bias')
    if (block_m not in {32,64} or block_n not in {64,128}
            or group_m not in {4,8,16,32} or num_stages not in {2,3}):
        raise ValueError('Unsupported implicit temporal tile')
    if low_count.shape!=() or low_count.device!=value.device or low_count.dtype!=torch.int64:
        raise ValueError('Expected refreshed device low-weight count')
    rows=batch*frames*height*width
    output=empty_conv_buffer('temporal',(batch,outputs,frames,height,width),dtype=torch.float32,
                        device=value.device)
    if debug_poison:output.fill_(float('nan'))
    eligible=(batch==1 and frames==16 and outputs==channels
        and (channels,height,width) in {(320,40,64),(640,20,32)}
        and block_m==64 and block_n==64 and num_stages==2
        and group_m==(8 if channels==320 else 16)
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cuda.matmul.allow_tf32)
    if input_parts is not None:
        assert not interleaved and len(input_parts)==2
        assert all(p.dtype==torch.float16 and p.device==value.device and p.numel()==value.numel() and p.is_contiguous() for p in input_parts)
        packed=tuple(input_parts);guards=packed;pack_kernel=None
    elif interleaved:
        storage=empty_conv_buffer('temporal',value.numel()+128,device=value.device,dtype=torch.int32)
        storage.fill_(0x5a5a5a5a) if debug_poison else None
        packed=(storage[64:-64],storage[64:-64]);guards=(storage,)
    else:
        guards=tuple(empty_conv_buffer('temporal',value.numel()+128,device=value.device,dtype=torch.float16) for _ in range(2))
        for storage in guards:
            if debug_poison:storage.fill_(float('nan'))
        packed=tuple(storage[64:-64] for storage in guards)
    if input_parts is None:
        pack_kernel=_split_ncthw[(triton.cdiv(value.numel(),1024),)](
            value,*packed,value.numel(),interleaved,1024,num_warps=4,enable_fp_fusion=False)
    kernels=[]
    for three in [False,True]:
        tuned=schedule=='3' and eligible and not three
        selected_group=32 if tuned and channels==320 else group_m
        selected_stages=2 if tuned else num_stages
        compiled=_presplit_temporal[(triton.cdiv(rows,block_m)*triton.cdiv(outputs,block_n),)](
            *packed,*parts,parts[0] if bias is None else bias,output,low_count,
            rows,outputs,channels,frames,height*width,bias is not None,
            block_m,block_n,64,selected_group,three,interleaved,num_warps=4,num_stages=selected_stages,
            enable_fp_fusion=False)
        kernels.append(compiled)
    return (output,kernels,pack_kernel,guards) if return_kernels else output

from unifolm_wma.utils.triton_split_group_norm import _partial,_finish_stats
from unifolm_wma.utils import norm_buffer_allocation as _norm_allocation

@triton.jit
def _norm_split_apply(X, Weight, Bias, Moments, High, Low, SIZE: tl.constexpr,
           GROUPS: tl.constexpr, CHANNELS: tl.constexpr,
           FRAMES: tl.constexpr, HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
           S0: tl.constexpr, S1: tl.constexpr, ST: tl.constexpr,
           SH: tl.constexpr, SW: tl.constexpr, HALF: tl.constexpr,
           SILU: tl.constexpr,
           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = off < SIZE
    spatial = FRAMES * HEIGHT * WIDTH
    channel = (row % GROUPS) * (CHANNELS // GROUPS) + off // spatial
    rem = off % spatial
    position = ((row // GROUPS) * S0 + channel * S1 +
                (rem // (HEIGHT * WIDTH)) * ST +
                ((rem // WIDTH) % HEIGHT) * SH + (rem % WIDTH) * SW)
    value = tl.load(X + position, valid, other=0).to(tl.float32)
    mean = tl.load(Moments + row * 2).to(tl.float32)
    inverse = tl.load(Moments + row * 2 + 1).to(tl.float32)
    weight = tl.load(Weight + channel, valid, other=0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0).to(tl.float32)
    scale = inverse * weight
    shift = tl.math.fma(-scale, mean, bias)
    normalized = tl.math.fma(scale, value, shift)
    if HALF:
        normalized = normalized.to(tl.float16).to(tl.float32)
    result = normalized
    if SILU:
        result = normalized / (1. + tl.exp(-normalized))
    high,low=_half_split(result,1.)
    tl.store(High+row*SIZE+off,high,valid)
    tl.store(Low+row*SIZE+off,low,valid)


def norm_presplit_temporal_conv(value,norm,parts,count,bias=None,*,debug_poison=False,return_kernels=False):
    assert value.is_cuda and value.dtype==torch.float32 and value.ndim==5
    assert not torch.is_grad_enabled() and not norm.training
    batch,channels,frames,height,width=value.shape
    assert norm.num_channels==channels and channels%norm.num_groups==0
    assert all(p is not None and p.dtype==torch.float32 and p.is_contiguous() and p.device==value.device for p in [norm.weight,norm.bias])
    groups=norm.num_groups;size=channels//groups*frames*height*width
    tile=4096;number=triton.cdiv(size,tile)
    stats=_norm_allocation.empty('group',(batch*groups,number,2),device=value.device,dtype=torch.float32)
    moments=_norm_allocation.empty('group',(batch*groups,2),device=value.device,dtype=torch.float32)
    guards=tuple(_norm_allocation.empty('group',value.numel()+128,device=value.device,dtype=torch.float16) for _ in range(2))
    packed=tuple(p[64:-64] for p in guards)
    if debug_poison:
        for p in [stats,moments,*guards]:p.fill_(float('nan'))
    geometry=(groups,channels,frames,height,width,*value.stride())
    _partial[(batch*groups,number)](value,stats,size,number,*geometry,tile,num_warps=4,enable_fp_fusion=False)
    _finish_stats[(batch*groups,)](stats,moments,size,number,norm.eps,False,tile,
        triton.next_power_of_2(number),num_warps=4,enable_fp_fusion=False)
    apply=_norm_split_apply[(batch*groups,triton.cdiv(size,1024))](value,norm.weight,norm.bias,moments,
        *packed,size,*geometry,False,True,1024,num_warps=4,enable_fp_fusion=False)
    group=8 if channels==320 else 16 if channels==640 else 32
    output,kernels,_,_=presplit_temporal_conv(value,parts,count,bias,group_m=group,
        debug_poison=debug_poison,return_kernels=True,input_parts=packed)
    return (output,kernels,apply,guards,stats,moments) if return_kernels else output
