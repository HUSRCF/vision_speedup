"""Retain FP16 three-product blocks; compensate FP32 summation across blocks."""
import triton as tr
import triton.language as tl


@tr.jit
def boundary(value):
    return tl.inline_asm_elementwise('mov.b32 $0, $1;',constraints='=f,f',
        args=[value],dtype=tl.float32,is_pure=True,pack=1)


@tr.jit
def transformed_gemm_fp16_kahan(W,X,Y,ROWS:tl.constexpr,COLS:tl.constexpr,K:tl.constexpr,
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
        w=tl.load(W+batch*ROWS*K+rows[:,None]*K+k[None,:],
                  (rows[:,None]<ROWS)&(k[None,:]<K),other=0.)
        x=tl.load(X+batch*K*COLS+k[:,None]*COLS+cols[None,:],
                  (k[:,None]<K)&(cols[None,:]<COLS),other=0.)
        wh=w.to(tl.float16);xh=x.to(tl.float16)
        wl=((w-wh.to(tl.float32))*2048.).to(tl.float16)
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
