"""Prototype device-predicated third product for the vendor chunked FFN."""
import ctypes
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary


@triton.jit
def _third_product(XH,WL,Correction,Active,M:tl.constexpr,N:tl.constexpr,
                   K:tl.constexpr,CHUNK:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    if tl.load(Active)!=0:
        pid,chunk=tl.program_id(0),tl.program_id(1)
        rows=(pid//tl.cdiv(N,BN))*BM+tl.arange(0,BM)
        cols=(pid%tl.cdiv(N,BN))*BN+tl.arange(0,BN)
        inner=tl.arange(0,64)
        acc=tl.full((BM,BN),0.,tl.float32)
        for block in range(tl.cdiv(CHUNK,64)):
            k=chunk*CHUNK+block*64+inner
            valid_k=(k<K)&(k<(chunk+1)*CHUNK)
            xh=tl.load(XH+rows[:,None]*K+k[None,:],(rows[:,None]<M)&valid_k[None,:],other=0)
            wl=tl.load(WL+cols[None,:]*K+k[:,None],(cols[None,:]<N)&valid_k[:,None],other=0)
            acc=tl.dot(xh,wl,acc)
        offset=chunk*M*N+rows[:,None]*N+cols[None,:]
        valid=(rows[:,None]<M)&(cols[None,:]<N)
        previous=tl.load(Correction+offset,valid,other=0)
        updated=_rounding_boundary(acc)+previous
        tl.store(Correction+offset,updated,valid)


def device_alpha_product(lib,handle,x,w,out,chunk,alpha,one):
    mode=ctypes.c_int()
    assert lib.cublasGetPointerMode_v2(handle,ctypes.byref(mode))==0 and mode.value==0
    rows,inner=x.shape;columns=w.shape[0]
    assert inner%chunk==0 and chunk%64==0
    _third_product[(triton.cdiv(rows,64)*triton.cdiv(columns,64),inner//chunk)](
        x,w,out,alpha,rows,columns,inner,chunk,64,64,
        num_warps=4,num_stages=3,enable_fp_fusion=False)
    assert lib.cublasGetPointerMode_v2(handle,ctypes.byref(mode))==0 and mode.value==0
