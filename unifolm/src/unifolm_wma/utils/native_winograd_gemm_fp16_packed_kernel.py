"""Prepack only constant native filter matrices; retain FP16x3 block arithmetic."""
import triton as tr
import triton.language as tl
from .native_winograd_gemm_fp16_kernel import boundary


@tr.jit
def pack_native_filter(W,WH,WL,N:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    w=tl.load(W+index,index<N,other=0.)
    wh=w.to(tl.float16)
    wl=((w-wh.to(tl.float32))*2048.).to(tl.float16)
    tl.store(WH+index,wh,index<N)
    tl.store(WL+index,wl,index<N)


@tr.jit
def transformed_gemm_fp16_packed(WH,WL,X,Y,ROWS:tl.constexpr,COLS:tl.constexpr,K:tl.constexpr,
                                 BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row_tile=tl.program_id(0)//tl.cdiv(COLS,BN)
    col_tile=tl.program_id(0)%tl.cdiv(COLS,BN)
    batch=tl.program_id(1)
    rows=row_tile*BM+tl.arange(0,BM)
    cols=col_tile*BN+tl.arange(0,BN)
    inner=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=block*BK+inner
        offset=batch*ROWS*K+rows[:,None]*K+k[None,:]
        mask=(rows[:,None]<ROWS)&(k[None,:]<K)
        wh=tl.load(WH+offset,mask,other=0.)
        wl=tl.load(WL+offset,mask,other=0.)
        x=tl.load(X+batch*K*COLS+k[:,None]*COLS+cols[None,:],
                  (k[:,None]<K)&(cols[None,:]<COLS),other=0.)
        xh=x.to(tl.float16)
        xl=((x-xh.to(tl.float32))*2048.).to(tl.float16)
        primary=boundary(tl.dot(wh,xh,allow_tf32=False))
        correction=tl.dot(wl,xh,allow_tf32=False)
        correction=tl.dot(wh,xl,correction,allow_tf32=False)
        acc+=boundary(primary+correction*(1./2048.))
    tl.store(Y+batch*ROWS*COLS+rows[:,None]*COLS+cols[None,:],acc,
             (rows[:,None]<ROWS)&(cols[None,:]<COLS))
