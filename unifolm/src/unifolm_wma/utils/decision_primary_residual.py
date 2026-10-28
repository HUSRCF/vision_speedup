import ctypes
import torch
import triton
import triton.language as tl
from .cublas_compensated_linear import _library,_READY,_stream_key,_gemm

@triton.jit
def _finish_residual(Primary,Bias,Residual,SIZE:tl.constexpr,N:tl.constexpr,
             HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<SIZE
    result=tl.load(Primary+i,valid,other=0.)*(1./256.)
    if HAS_BIAS:result+=tl.load(Bias+i%N,valid,other=0.)
    result+=tl.load(Residual+i,valid,other=0.)
    tl.store(Primary+i,result,valid)


def primary_linear_residual(value,high_weight,residual,bias=None,*,debug_poison=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
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
        # Dynamic inputs are converted on every invocation/replay, inside timing.
        xh=value.view(rows,inner).to(torch.float16)
        primary=torch.empty((rows,columns),device=value.device,dtype=torch.float32)
        if debug_poison:primary.fill_(float('nan'))
        _gemm(lib,handle,xh,high_weight,primary)
        _finish_residual[(triton.cdiv(rows*columns,256),)](primary,
            high_weight if bias is None else bias,residual,rows*columns,columns,bias is not None,
            256,num_warps=4,enable_fp_fusion=False)
        return primary.view(*value.shape[:-1],columns)

