import ctypes
import torch
import triton
import triton.language as tl
from .triton_prepacked_geglu import _short_product
from .triton_compensated_linear import _rounding_boundary
from .geglu_output_allocation import empty as empty_geglu
from .cublas_compensated_linear import _library,_READY,_stream_key,_gemm
from .decision_primary_residual import _finish_residual

@triton.jit
def _primary_geglu_hidden_high(XH,XL,WH,WL,Bias,Y,M:tl.constexpr,C:tl.constexpr,
                   K:tl.constexpr,HAS_BIAS:tl.constexpr,BM:tl.constexpr,
                   BN:tl.constexpr,GROUP_M:tl.constexpr,PRIMARY_GATE:tl.constexpr):
    pid=tl.program_id(0)
    row_tiles,col_tiles=tl.cdiv(M,BM),tl.cdiv(C,BN)
    group=pid//(GROUP_M*col_tiles)
    first_row=group*GROUP_M
    group_rows=tl.minimum(GROUP_M,row_tiles-first_row)
    local=pid%(GROUP_M*col_tiles)
    rows=(first_row+local%group_rows)*BM+tl.arange(0,BM)
    cols=(local//group_rows)*BN+tl.arange(0,BN)
    inner=tl.arange(0,64)
    value=tl.full((BM,BN),0.,tl.float32)
    gate=tl.full((BM,BN),0.,tl.float32)
    for start in range(tl.cdiv(K,64)):
        k=start*64+inner
        xoffset=rows[:,None]*K+k[None,:]
        xvalid=(rows[:,None]<M)&(k[None,:]<K)
        xh=tl.load(XH+xoffset,xvalid,other=0.)
        offset=cols[None,:]*K+k[:,None]
        valid=(cols[None,:]<C)&(k[:,None]<K)
        vh=tl.load(WH+offset,valid,other=0.)
        gh=tl.load(WH+offset+C*K,valid,other=0.)
        value=tl.dot(xh,vh,value)
        if PRIMARY_GATE:
            gate=tl.dot(xh,gh,gate)
        else:
            xl=tl.load(XL+xoffset,xvalid,other=0.)
            gl=tl.load(WL+offset+C*K,valid,other=0.)
            gate+=_short_product(xh,xl,gh,gl)
    value=value*(1./256.)
    if PRIMARY_GATE:gate=gate*(1./256.)
    if HAS_BIAS:
        value+=tl.load(Bias+cols,cols<C,other=0.)[None,:]
        gate+=tl.load(Bias+cols+C,cols<C,other=0.)[None,:]
    gelu=(gate*.5)*(1.+tl.math.erf(gate*.7071067811865476))
    hidden=_rounding_boundary(value*gelu)
    tl.store(Y+rows[:,None]*C+cols[None,:],hidden.to(tl.float16),
             (rows[:,None]<M)&(cols[None,:]<C))


def primary_half_input_residual(value,high_weight,residual,bias=None,*,debug_poison=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float16
            or value.ndim<1 or not value.is_contiguous() or high_weight.ndim!=2
            or high_weight.dtype!=torch.float16 or high_weight.device!=value.device
            or not high_weight.is_contiguous() or value.shape[-1]!=high_weight.shape[1]):
        raise ValueError('Expected contiguous FP32 input and compensated high Half weights')
    rows=value.numel()//value.shape[-1]
    columns,inner=high_weight.shape
    if min(rows,columns,inner)<=0:raise ValueError('Empty GEMM unsupported')
    if bias is not None and (bias.shape!=(columns,) or bias.dtype!=torch.float32
            or bias.device!=value.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    if (residual.shape!=(*value.shape[:-1],columns) or residual.device!=value.device
            or residual.dtype!=torch.float32 or not residual.is_contiguous()):
        raise ValueError('Invalid original FP32 residual')
    with torch.cuda.device(value.device):
        key=_stream_key(value.device)
        if torch.cuda.is_current_stream_capturing() and key not in _READY:return None
        lib=_library()
        handle=ctypes.c_void_p(torch.cuda.current_blas_handle())
        mode=ctypes.c_int()
        status=lib.cublasGetPointerMode_v2(handle,ctypes.byref(mode))
        if status or mode.value!=0:raise RuntimeError('cuBLAS host pointer mode required')
        _READY.add(key)
        # Fresh Half inputs are produced in the preceding timed GEGLU kernel.
        xh=value.view(rows,inner)
        primary=torch.empty((rows,columns),device=value.device,dtype=torch.float32)
        if debug_poison:primary.fill_(float('nan'))
        _gemm(lib,handle,xh,high_weight,primary)
        _finish_residual[(triton.cdiv(rows*columns,256),)](primary,
            high_weight if bias is None else bias,residual,rows*columns,columns,bias is not None,
            256,num_warps=4,enable_fp_fusion=False)
        return primary.view(*value.shape[:-1],columns)



def primary_geglu_hidden_high(inputs,parts,bias=None):
    high,low=inputs
    assert high.is_cuda and high.ndim==2 and high.dtype==low.dtype==torch.float16 and high.shape==low.shape
    rows,inner=high.shape;channels=parts[0].shape[0]//2
    assert inner==1280 and channels==5120 and len(parts)==2
    assert all(p.shape==(channels*2,inner) and p.dtype==torch.float16 and p.device==high.device and p.is_contiguous() for p in parts)
    output=empty_geglu((rows,channels),device=high.device,dtype=torch.float16)
    _primary_geglu_hidden_high[(triton.cdiv(rows,64)*triton.cdiv(channels,32),)](
        high,low,*parts,parts[0] if bias is None else bias,output,
        rows,channels,inner,bias is not None,64,32,32,True,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return output
