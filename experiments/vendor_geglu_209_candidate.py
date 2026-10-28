"""Numerical GEGLU: vendor batched high/low products, FP32 epilogue."""
import ctypes
import torch,triton
import triton.language as tl
from unifolm_wma.utils.cublas_chunked_compensated_linear import _batched_library,_batched_gemm
from unifolm_wma.utils.triton_predicated_chunked_correction import _third_product
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary
from unifolm_wma.utils.triton_half_compensated_linear import _half_split

@triton.jit
def finish(P,Corr,Bias,YH,YL,V,G,M:tl.constexpr,C:tl.constexpr,
           CHUNKS:tl.constexpr,BOTH:tl.constexpr,HAS_BIAS:tl.constexpr,
           DEBUG:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<M*C;row=i//C;col=i%C
    value=tl.full((BLOCK,),0.,tl.float32)
    gate=tl.full((BLOCK,),0.,tl.float32)
    for chunk in range(CHUNKS):
        pv=tl.load(P+chunk*M*(2*C)+row*(2*C)+col,valid,other=0.)
        pg=tl.load(P+chunk*M*(2*C)+row*(2*C)+col+C,valid,other=0.)
        if BOTH:
            cv=tl.load(Corr+chunk*M*(2*C)+row*(2*C)+col,valid,other=0.)
            cg=tl.load(Corr+chunk*M*(2*C)+row*(2*C)+col+C,valid,other=0.)
            gate+=_rounding_boundary(pg+cg*(1./2048.))*(1./256.)
        else:
            cv=tl.load(Corr+chunk*M*C+i,valid,other=0.)
            gate+=pg
        value+=_rounding_boundary(pv+cv*(1./2048.))*(1./256.)
    if not BOTH:gate*=1./256.
    if HAS_BIAS:
        value+=tl.load(Bias+col,valid,other=0.)
        gate+=tl.load(Bias+col+C,valid,other=0.)
    if DEBUG:
        tl.store(V+i,value,valid);tl.store(G+i,gate,valid)
    gelu=(gate*.5)*(1.+tl.math.erf(gate*.7071067811865476))
    hidden=_rounding_boundary(value*gelu)
    high,low=_half_split(hidden,1.)
    tl.store(YH+i,high,valid);tl.store(YL+i,low,valid)

def allocate(m,c,k,chunk,kind,device):
    assert k%chunk==0 and kind in ('primary','both')
    values=[]
    for columns in (2*c,2*c if kind=='both' else c):
        backing=torch.full(((k//chunk)*m*columns+128,),12345.,device=device,dtype=torch.float32)
        values.append(backing[64:-64].reshape(k//chunk,m,columns))
    return tuple(values)

def verify(buffers):
    for view in buffers:
        assert torch.isfinite(view).all()
        assert (view._base[:64]==12345.).all() and (view._base[-64:]==12345.).all()

def launch(inputs,weights,count,bias,outputs,buffers,kind,chunk,debug):
    assert not torch.is_grad_enabled()
    xh,xl=inputs;wh,wl=weights;m,k=xh.shape;c=wh.shape[0]//2
    assert xh.dtype==xl.dtype==wh.dtype==wl.dtype==torch.float16
    assert all(t.is_contiguous() for t in (*inputs,*weights)) and k%chunk==0
    primary,correction=buffers;columns=c*2 if kind=='both' else c
    lib=_batched_library();handle=ctypes.c_void_p(torch.cuda.current_blas_handle())
    mode=ctypes.c_int();assert lib.cublasGetPointerMode_v2(handle,ctypes.byref(mode))==0 and mode.value==0
    _batched_gemm(lib,handle,xh,wh,primary,chunk)
    _batched_gemm(lib,handle,xl,wh[:columns],correction,chunk)
    _third_product[(triton.cdiv(m,32)*triton.cdiv(columns,64),k//chunk)](
        xh,wl[:columns],correction,count,m,columns,k,chunk,32,64,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return finish[(triton.cdiv(m*c,256),)](primary,correction,wh if bias is None else bias,*outputs,
        m,c,k//chunk,kind=='both',bias is not None,debug,256,num_warps=4,enable_fp_fusion=False)
