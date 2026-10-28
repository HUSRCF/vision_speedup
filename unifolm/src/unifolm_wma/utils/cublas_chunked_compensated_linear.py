"""Diagnostic batched short-K vendor products with ordered FP32 reduction."""
import ctypes
from functools import lru_cache
import torch
from torch.utils import deterministic
import triton
import triton.language as tl
from .cublas_compensated_linear import _library,_READY,_stream_key,_ONE,_ZERO
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import pack_half_weight


@lru_cache(None)
def _batched_library():
    lib = _library()
    void,integer,long = ctypes.c_void_p,ctypes.c_int,ctypes.c_longlong
    lib.cublasGemmStridedBatchedEx.argtypes = [void,integer,integer,integer,integer,
        integer,void,void,integer,integer,long,void,integer,integer,long,
        void,void,integer,integer,long,integer,integer,integer]
    lib.cublasGemmStridedBatchedEx.restype = integer
    return lib


def _written_buffer(shape,device,debug_fill):
    # Every element is overwritten by beta=0 GEMM or the final Triton store.
    # Keep the allocation policy local and restore it even on allocation error.
    old = deterministic.fill_uninitialized_memory
    try:
        deterministic.fill_uninitialized_memory = False
        result = torch.empty(shape,device=device,dtype=torch.float32)
    finally:
        deterministic.fill_uninitialized_memory = old
    if debug_fill is not None:result.fill_(debug_fill)
    return result


def _batched_gemm(lib,handle,value,weight,output,chunk,*,accumulate=False):
    rows,inner = value.shape
    columns = weight.shape[0]
    status = lib.cublasGemmStridedBatchedEx(handle,1,0,columns,rows,chunk,
        ctypes.byref(_ONE),ctypes.c_void_p(weight.data_ptr()),2,inner,chunk,
        ctypes.c_void_p(value.data_ptr()),2,inner,chunk,
        ctypes.byref(_ONE if accumulate else _ZERO),ctypes.c_void_p(output.data_ptr()),
        0,columns,rows*columns,inner//chunk,68,-1)
    if status:raise RuntimeError(f'cublasGemmStridedBatchedEx failed with status {status}')


@triton.jit
def _reduce(Primary,Correction,Bias,Output,M:tl.constexpr,N:tl.constexpr,
             CHUNKS:tl.constexpr,HAS_BIAS:tl.constexpr,GROUPS:tl.constexpr,
             BLOCK:tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid = i<M*N
    acc = tl.full((BLOCK,),0.,tl.float32)
    for chunk in range(CHUNKS):
        p = tl.load(Primary+chunk*M*N+i,valid,other=0)
        c = tl.load(Correction+chunk*M*N+i,valid,other=0)
        block = _rounding_boundary(p+c*(1./2048.))
        acc += block*(1./256.)
    column = i%N
    if HAS_BIAS:acc += tl.load(Bias+column,valid,other=0)
    if GROUPS==1:
        output = i
    else:
        part_n = N//GROUPS
        output = (column//part_n)*M*part_n+(i//N)*part_n+column%part_n
    tl.store(Output+output,acc,valid)


def chunked_compensated_linear(value,parts,bias=None,*,chunk_k=256,
                                output_groups=1,debug_fill=None):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim<1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous CUDA FP32 inference input')
    if len(parts)!=2 or any(p.ndim!=2 or p.shape!=parts[0].shape
            or p.shape[1]!=value.shape[-1] or p.dtype!=torch.float16
            or p.device!=value.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid compensated Half weights')
    columns,inner = parts[0].shape
    rows = value.numel()//inner
    if min(rows,columns,inner)<=0 or max(rows,columns,inner)>=2**31:
        raise ValueError('Invalid GEMM dimensions')
    if not isinstance(chunk_k,int) or not 1<=chunk_k<=4096 or inner%chunk_k:
        raise ValueError('Chunk must be positive,<=4096 and divide reduction')
    if output_groups not in {1,3} or columns%output_groups:
        raise ValueError('Output groups must be1or3 and divide columns')
    if bias is not None and (bias.shape!=(columns,) or bias.device!=value.device
            or bias.dtype!=torch.float32 or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    with torch.cuda.device(value.device):
        key = _stream_key(value.device)
        if torch.cuda.is_current_stream_capturing() and key not in _READY:
            return None
        lib = _batched_library()
        handle = ctypes.c_void_p(torch.cuda.current_blas_handle())
        mode = ctypes.c_int()
        status = lib.cublasGetPointerMode_v2(handle,ctypes.byref(mode))
        if status or mode.value!=0:
            raise RuntimeError('cuBLAS host pointer mode required')
        _READY.add(key)
        xh,xl = pack_half_weight(value.view(rows,inner),weight_scale=1.)
        chunks = inner//chunk_k
        shape = (chunks,rows,columns)
        primary = _written_buffer(shape,value.device,debug_fill)
        correction = _written_buffer(shape,value.device,debug_fill)
        _batched_gemm(lib,handle,xh,parts[0],primary,chunk_k)
        _batched_gemm(lib,handle,xl,parts[0],correction,chunk_k)
        _batched_gemm(lib,handle,xh,parts[1],correction,chunk_k,accumulate=True)
        output_shape = ((*value.shape[:-1],columns) if output_groups==1 else
                        (output_groups,*value.shape[:-1],columns//output_groups))
        output = _written_buffer(output_shape,value.device,debug_fill)
        _reduce[(triton.cdiv(rows*columns,256),)](primary,correction,
            parts[0] if bias is None else bias,output,rows,columns,chunks,
            bias is not None,output_groups,256,num_warps=4,enable_fp_fusion=False)
        return output
