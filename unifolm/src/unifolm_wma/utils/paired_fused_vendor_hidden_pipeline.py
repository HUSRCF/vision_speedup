"""Row-paired first two GEMMs plus one fused correction/reduction epilogue."""
import ctypes
import torch
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .cublas_compensated_linear import _READY,_stream_key
from .cublas_chunked_compensated_linear import _batched_library,_batched_gemm,_written_buffer


@triton.jit
def _finish(XH,WL,Partial,Active,Bias,Residual,Output,
            M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,CHUNK:tl.constexpr,
            HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    pid=tl.program_id(0)
    if tl.load(Active)==0:
        i=pid*BLOCK+tl.arange(0,BLOCK);valid=i<M*N
        acc=tl.full((BLOCK,),0.,tl.float32)
        for chunk in range(tl.cdiv(K,CHUNK)):
            p=tl.load(Partial+chunk*2*M*N+i,valid,other=0.)
            c=tl.load(Partial+chunk*2*M*N+M*N+i,valid,other=0.)
            block=_rounding_boundary(p+c*(1./2048.))
            acc+=block*(1./256.)
        if HAS_BIAS:acc+=tl.load(Bias+i%N,valid,other=0.)
        acc=_rounding_boundary(acc)+tl.load(Residual+i,valid,other=0.)
        tl.store(Output+i,acc,valid)
    else:
        if pid<tl.cdiv(M,BM)*tl.cdiv(N,BN):
            rows=(pid//tl.cdiv(N,BN))*BM+tl.arange(0,BM)
            cols=(pid%tl.cdiv(N,BN))*BN+tl.arange(0,BN)
            inner=tl.arange(0,64)
            valid=(rows[:,None]<M)&(cols[None,:]<N)
            offset=rows[:,None]*N+cols[None,:]
            total=tl.full((BM,BN),0.,tl.float32)
            for chunk in range(tl.cdiv(K,CHUNK)):
                acc=tl.full((BM,BN),0.,tl.float32)
                for block in range(tl.cdiv(CHUNK,64)):
                    k=chunk*CHUNK+block*64+inner
                    valid_k=(k<K)&(k<(chunk+1)*CHUNK)
                    xh=tl.load(XH+rows[:,None]*K+k[None,:],(rows[:,None]<M)&valid_k[None,:],other=0.)
                    wl=tl.load(WL+cols[None,:]*K+k[:,None],(cols[None,:]<N)&valid_k[:,None],other=0.)
                    acc=tl.dot(xh,wl,acc)
                p=tl.load(Partial+chunk*2*M*N+offset,valid,other=0.)
                c=tl.load(Partial+chunk*2*M*N+M*N+offset,valid,other=0.)
                updated=_rounding_boundary(acc)+c
                reduced=_rounding_boundary(p+updated*(1./2048.))
                total+=reduced*(1./256.)
            if HAS_BIAS:total+=tl.load(Bias+cols,cols<N,other=0.)[None,:]
            total=_rounding_boundary(total)+tl.load(Residual+offset,valid,other=0.)
            tl.store(Output+offset,total,valid)


def paired_hidden_residual(stack,parts,residual,count,bias=None,*,chunk_k=512,debug_poison=False):
    if (torch.is_grad_enabled() or not stack.is_cuda or stack.dtype!=torch.float16
            or stack.ndim!=3 or stack.shape[0]!=2 or not stack.is_contiguous()):
        raise ValueError('Expected two contiguous planes of fresh Half inputs')
    _,rows,inner=stack.shape
    value=stack
    if len(parts)!=2 or any(p.dtype!=torch.float16 or p.ndim!=2 or p.shape!=parts[0].shape
            or p.shape[1]!=inner or p.device!=stack.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid compensated weight parts')
    columns=parts[0].shape[0]
    if min(rows,inner,columns)<=0 or inner%chunk_k or chunk_k%64:raise ValueError('Unsupported chunk reduction')
    if residual.shape!=(rows,columns) or residual.dtype!=torch.float32 or residual.device!=value.device or not residual.is_contiguous():
        raise ValueError('Invalid original FP32 residual')
    if count.shape!=() or count.dtype!=torch.int64 or count.device!=value.device:raise ValueError('Invalid device predicate')
    if bias is not None and (bias.shape!=(columns,) or bias.dtype!=torch.float32 or bias.device!=value.device or not bias.is_contiguous()):
        raise ValueError('Invalid original FP32 bias')
    key=_stream_key(value.device)
    if torch.cuda.is_current_stream_capturing() and key not in _READY:return None
    lib=_batched_library();handle=ctypes.c_void_p(torch.cuda.current_blas_handle());mode=ctypes.c_int()
    if lib.cublasGetPointerMode_v2(handle,ctypes.byref(mode)) or mode.value!=0:raise RuntimeError('cuBLAS host pointer mode required')
    _READY.add(key)
    poison=float('nan') if debug_poison else None
    partial=_written_buffer((inner//chunk_k,2*rows,columns),value.device,poison)
    _batched_gemm(lib,handle,stack.view(2*rows,inner),parts[0],partial,chunk_k)
    output=_written_buffer((rows,columns),value.device,poison)
    blocks=max(triton.cdiv(rows*columns,256),triton.cdiv(rows,16)*triton.cdiv(columns,32))
    _finish[(blocks,)](stack,parts[1],partial,count,parts[0] if bias is None else bias,residual,
        output,rows,columns,inner,chunk_k,bias is not None,256,16,32,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return output
