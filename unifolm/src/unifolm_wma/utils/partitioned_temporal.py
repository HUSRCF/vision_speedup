"""Fuse temporal indexing and input splitting into the accepted GEMM math."""
import torch
from .conv_buffer_allocation import empty as empty_conv_buffer
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary


@triton.jit
def _partitioned_temporal(X, WH, WL, Bias, Output, LowCount,
                       M: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
                       T: tl.constexpr, S: tl.constexpr,
                       HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, THREE: tl.constexpr):
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
            value=tl.load(X+source,valid,other=0)
            xh,xl=_half_split(value,1.)
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


def partitioned_temporal_conv(value,parts,low_count,bias=None,*,block_m=64,block_n=64,
                            group_m=32,num_stages=2,debug_poison=False,return_kernels=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim!=5 or not value.is_contiguous()):
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
    kernels=[]
    for three in [False,True]:
        tuned=schedule=='3' and eligible and not three
        selected_group=32 if tuned and channels==320 else group_m
        selected_stages=3 if tuned else num_stages
        compiled=_partitioned_temporal[(triton.cdiv(rows,block_m)*triton.cdiv(outputs,block_n),)](
            value,*parts,parts[0] if bias is None else bias,output,low_count,
            rows,outputs,channels,frames,height*width,bias is not None,
            block_m,block_n,64,selected_group,three,num_warps=4,num_stages=selected_stages,
            enable_fp_fusion=False)
        kernels.append(compiled)
    return (output,kernels) if return_kernels else output
