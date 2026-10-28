"""Diagnostic three-product Half GEMM with vendor full-K FP32 accumulators.

The long reduction differs from the accepted short-BK Triton arithmetic.
Inference installation requires separate stage guards and full video checks.
API constants/signature were checked against the installed CUDA12.1 headers.
PyTorch's current handle owns stream/workspace setup; no handle state is set.
"""
import ctypes
from functools import lru_cache
from pathlib import Path
import site
import threading
import torch
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import pack_half_weight

_READY = set()
_ONE = ctypes.c_float(1.)
_ZERO = ctypes.c_float(0.)


@lru_cache(None)
def _library():
    path = next((Path(root)/'nvidia/cublas/lib/libcublas.so.12'
                 for root in site.getsitepackages()
                 if (Path(root)/'nvidia/cublas/lib/libcublas.so.12').is_file()),None)
    if path is None:raise RuntimeError('CUDA12 cuBLAS library was not found')
    lib = ctypes.CDLL(str(path))
    void,integer = ctypes.c_void_p,ctypes.c_int
    lib.cublasGemmEx.argtypes = [void,integer,integer,integer,integer,integer,
        void,void,integer,integer,void,integer,integer,void,void,integer,
        integer,integer,integer]
    lib.cublasGemmEx.restype = integer
    lib.cublasGetPointerMode_v2.argtypes = [void,ctypes.POINTER(integer)]
    lib.cublasGetPointerMode_v2.restype = integer
    return lib


def _stream_key(device):
    return device.index,int(torch.cuda.current_stream(device).cuda_stream),threading.get_ident()


def _gemm(lib,handle,value,weight,output,*,accumulate=False):
    # Row-major X[M,K] @ W[N,K].T is column-major W.T[N,K] @ X.T[K,M].
    rows,inner = value.shape
    columns = weight.shape[0]
    status = lib.cublasGemmEx(handle,1,0,columns,rows,inner,
        ctypes.byref(_ONE),ctypes.c_void_p(weight.data_ptr()),2,inner,
        ctypes.c_void_p(value.data_ptr()),2,inner,
        ctypes.byref(_ONE if accumulate else _ZERO),ctypes.c_void_p(output.data_ptr()),
        0,columns,68,-1)
    if status:raise RuntimeError(f'cublasGemmEx failed with status {status}')


@triton.jit
def _combine(Primary,Correction,Bias,Output,SIZE:tl.constexpr,M:tl.constexpr,
              N:tl.constexpr,HAS_BIAS:tl.constexpr,GROUPS:tl.constexpr,BLOCK:tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid = i<SIZE
    primary = tl.load(Primary+i,valid,other=0)
    correction = tl.load(Correction+i,valid,other=0)
    value = _rounding_boundary(primary+correction*(1./2048.))*(1./256.)
    column = i%N
    if HAS_BIAS:value += tl.load(Bias+column,valid,other=0)
    if GROUPS==1:
        output = i
    else:
        part_n = N//GROUPS
        output = (column//part_n)*M*part_n+(i//N)*part_n+column%part_n
    tl.store(Output+output,value,valid)


def cublas_compensated_linear(value,parts,bias=None,*,output_groups=1):
    """Return None for an unwarmed stream during capture; callers must fall back."""
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim<1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous CUDA FP32 inference input')
    if len(parts)!=2 or any(p.ndim!=2 or p.shape!=parts[0].shape
            or p.shape[1]!=value.shape[-1] or p.device!=value.device
            or p.dtype!=torch.float16 or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid compensated Half weights')
    columns,inner = parts[0].shape
    rows = value.numel()//inner
    if min(rows,columns,inner)<=0 or max(rows,columns,inner)>=2**31:
        raise ValueError('Invalid GEMM dimensions')
    if output_groups not in {1,3} or columns%output_groups:
        raise ValueError('Output groups must be1or3 and divide columns')
    if bias is not None and (bias.shape!=(columns,) or bias.device!=value.device
            or bias.dtype!=torch.float32 or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    with torch.cuda.device(value.device):
        key = _stream_key(value.device)
        if torch.cuda.is_current_stream_capturing() and key not in _READY:
            return None
        lib = _library()
        handle = ctypes.c_void_p(torch.cuda.current_blas_handle())
        pointer_mode = ctypes.c_int()
        status = lib.cublasGetPointerMode_v2(handle,ctypes.byref(pointer_mode))
        if status or pointer_mode.value!=0:
            raise RuntimeError(f'cuBLAS host pointer mode required,status={status},mode={pointer_mode.value}')
        _READY.add(key)
        xh,xl = pack_half_weight(value.view(rows,inner),weight_scale=1.)
        primary = torch.empty((rows,columns),device=value.device,dtype=torch.float32)
        correction = torch.empty_like(primary)
        _gemm(lib,handle,xh,parts[0],primary)
        _gemm(lib,handle,xl,parts[0],correction)
        _gemm(lib,handle,xh,parts[1],correction,accumulate=True)
        if output_groups==1:
            output = primary.view(*value.shape[:-1],columns)
        else:
            output = torch.empty((output_groups,*value.shape[:-1],columns//output_groups),
                                 device=value.device,dtype=torch.float32)
        _combine[(triton.cdiv(rows*columns,256),)](primary,correction,
            parts[0] if bias is None else bias,output,rows*columns,rows,columns,
            bias is not None,output_groups,256,num_warps=4,enable_fp_fusion=False)
        return output
