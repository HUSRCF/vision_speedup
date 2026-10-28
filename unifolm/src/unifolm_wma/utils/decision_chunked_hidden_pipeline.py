import ctypes
import torch
import triton
import triton.language as tl
from .triton_primary_geglu import _short_product
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary
from .geglu_output_allocation import empty as empty_geglu
from .cublas_compensated_linear import _READY,_stream_key
from .cublas_chunked_compensated_linear import _batched_library,_batched_gemm,_written_buffer

@triton.jit
def _primary_geglu_hidden_parts(XH,XL,WH,WL,Bias,YH,YL,M:tl.constexpr,C:tl.constexpr,
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
    high,low=_half_split(hidden,1.)
    valid=(rows[:,None]<M)&(cols[None,:]<C)
    tl.store(YH+rows[:,None]*C+cols[None,:],high,valid)
    tl.store(YL+rows[:,None]*C+cols[None,:],low,valid)


@triton.jit
def _reduce_residual(Primary,Correction,Bias,Residual,Output,M:tl.constexpr,N:tl.constexpr,
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
    acc=_rounding_boundary(acc)+tl.load(Residual+i,valid,other=0.)
    if GROUPS==1:
        output = i
    else:
        part_n = N//GROUPS
        output = (column//part_n)*M*part_n+(i//N)*part_n+column%part_n
    tl.store(Output+output,acc,valid)


def chunked_hidden_residual(inputs,parts,residual,bias=None,*,chunk_k=512,
                                output_groups=1,debug_fill=None):
    value=inputs[0]
    if len(inputs)!=2 or any(p.dtype!=torch.float16 or p.shape!=value.shape or p.device!=value.device
            or p.ndim!=2 or not p.is_contiguous() for p in inputs):
        raise ValueError('Expected fresh Half input parts')
    if residual.dtype!=torch.float32 or residual.device!=value.device or not residual.is_contiguous():
        raise ValueError('Expected original contiguous FP32 residual')
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float16
            or value.ndim<1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous CUDA FP32 inference input')
    if len(parts)!=2 or any(p.ndim!=2 or p.shape!=parts[0].shape
            or p.shape[1]!=value.shape[-1] or p.dtype!=torch.float16
            or p.device!=value.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid compensated Half weights')
    columns,inner = parts[0].shape
    rows = value.numel()//inner
    if residual.shape!=(rows,columns) or output_groups!=1:
        raise ValueError('Expected matching FP32 residual and one output group')
    if min(rows,columns,inner)<=0 or max(rows,columns,inner)>=2**31:
        raise ValueError('Invalid GEMM dimensions')
    if not isinstance(chunk_k,int) or not 1<=chunk_k<=4096 or inner%chunk_k:
        raise ValueError('Chunk must be positive,<=4096 and divide reduction')
    if output_groups not in {1,3} or columns%output_groups:
        raise ValueError('Output groups must be1or3 and divide columns')
    if bias is not None and (bias.shape!=(columns,) or bias.device!=value.device
            or bias.dtype!=torch.float32 or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    import os
    correction_mode=os.environ.get('UNIFOLM_DECISION_VENDOR_CORRECTION','0')
    if correction_mode not in {'0','1'}:
        raise ValueError('Invalid decision vendor correction mode')
    correction_count=None
    if (correction_mode=='1' and (rows,columns,inner,chunk_k,output_groups)==(10240,640,2560,512,1)
            and torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32):
        from .vendor_correction_count import get_count
        correction_count=get_count(parts)
    paired_mode=os.environ.get('UNIFOLM_DECISION_PAIRED_FUSED_VENDOR','0')
    if paired_mode not in {'0','1'}:
        raise ValueError('Invalid decision paired vendor mode')
    if paired_mode=='1' and correction_count is not None:
        stack=getattr(inputs[0],'_paired_vendor_stack',None)
        if (isinstance(stack,torch.Tensor) and stack.shape==(2,rows,inner)
                and stack.dtype==torch.float16 and stack.device==value.device and stack.is_contiguous()
                and stack.data_ptr()==inputs[0].data_ptr()
                and inputs[1].data_ptr()==stack.data_ptr()+rows*inner*stack.element_size()):
            from .paired_fused_vendor_hidden_pipeline import paired_hidden_residual
            with torch.cuda.device(value.device):
                return paired_hidden_residual(stack,parts,residual,correction_count,bias,
                    chunk_k=chunk_k,debug_poison=debug_fill is not None)
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
        xh,xl = inputs
        chunks = inner//chunk_k
        shape = (chunks,rows,columns)
        primary = _written_buffer(shape,value.device,debug_fill)
        correction = _written_buffer(shape,value.device,debug_fill)
        _batched_gemm(lib,handle,xh,parts[0],primary,chunk_k)
        _batched_gemm(lib,handle,xl,parts[0],correction,chunk_k)
        if correction_count is None:
            _batched_gemm(lib,handle,xh,parts[1],correction,chunk_k,accumulate=True)
        else:
            from .triton_predicated_chunked_correction import _third_product
            _third_product[(triton.cdiv(rows,64)*triton.cdiv(columns,64),chunks)](
                xh,parts[1],correction,correction_count,rows,columns,inner,chunk_k,64,64,
                num_warps=4,num_stages=3,enable_fp_fusion=False)
        output_shape = ((*value.shape[:-1],columns) if output_groups==1 else
                        (output_groups,*value.shape[:-1],columns//output_groups))
        output = _written_buffer(output_shape,value.device,debug_fill)
        _reduce_residual[(triton.cdiv(rows*columns,256),)](primary,correction,
            parts[0] if bias is None else bias,residual,output,rows,columns,chunks,
            bias is not None,output_groups,256,num_warps=4,enable_fp_fusion=False)
        return output



def primary_geglu_hidden_parts(inputs,parts,bias=None):
    high,low=inputs
    assert high.is_cuda and high.ndim==2 and high.dtype==low.dtype==torch.float16 and high.shape==low.shape
    rows,inner=high.shape;channels=parts[0].shape[0]//2
    assert inner==640 and channels==2560 and len(parts)==2
    assert all(p.shape==(channels*2,inner) and p.dtype==torch.float16 and p.device==high.device and p.is_contiguous() for p in parts)
    import os
    paired_mode=os.environ.get('UNIFOLM_DECISION_PAIRED_FUSED_VENDOR','0')
    if paired_mode not in {'0','1'}:
        raise ValueError('Invalid decision paired vendor mode')
    paired_selected=(paired_mode=='1' and rows==10240
        and os.environ.get('UNIFOLM_DECISION_VENDOR_CORRECTION','0')=='1'
        and torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
    if paired_selected:
        stack=empty_geglu((2,rows,channels),device=high.device,dtype=torch.float16)
        outputs=tuple(stack.unbind(0));outputs[0]._paired_vendor_stack=stack
    else:
        outputs=tuple(empty_geglu((rows,channels),device=high.device,dtype=torch.float16) for _ in range(2))
    _primary_geglu_hidden_parts[(triton.cdiv(rows,64)*triton.cdiv(channels,64),)](
        high,low,*parts,parts[0] if bias is None else bias,*outputs,
        rows,channels,inner,bias is not None,64,64,32,True,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return outputs
