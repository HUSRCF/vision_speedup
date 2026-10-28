"""Prepacked filter operands with the unchanged three-product Kahan block sum."""
import triton as tr
import triton.language as tl
from unifolm_wma.utils.native_winograd_gemm_fp16_kahan_kernel import boundary


@tr.jit
def transformed_gemm_fp16_kahan_packed(WH,WL,X,Y,ROWS:tl.constexpr,COLS:tl.constexpr,K:tl.constexpr,
                                     BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row_tile=tl.program_id(0)//tl.cdiv(COLS,BN)
    col_tile=tl.program_id(0)%tl.cdiv(COLS,BN)
    batch=tl.program_id(1)
    rows=row_tile*BM+tl.arange(0,BM)
    cols=col_tile*BN+tl.arange(0,BN)
    inner=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    compensation=tl.zeros((BM,BN),tl.float32)
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
        term=boundary(primary+correction*(1./2048.))
        adjusted=boundary(term-compensation)
        updated=boundary(acc+adjusted)
        compensation=boundary(boundary(updated-acc)-adjusted)
        acc=updated
    tl.store(Y+batch*ROWS*COLS+rows[:,None]*COLS+cols[None,:],acc,
             (rows[:,None]<ROWS)&(cols[None,:]<COLS))
