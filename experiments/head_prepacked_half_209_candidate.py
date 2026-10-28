"""Scaled RNA FP16 Conv1d: timed im2col plus split-K GEMM and reduction."""
import torch,triton
import triton.language as tl
from head_scaled_half_209_candidate import rounded,pack
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary

@triton.jit
def im2col(X,A,B:tl.constexpr,CI:tl.constexpr,L:tl.constexpr,KS:tl.constexpr,
           OL:tl.constexpr,STRIDE:tl.constexpr,PAD:tl.constexpr,DIL:tl.constexpr,
           BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    k=off%(CI*KS);row=off//(CI*KS)
    loc=(row%OL)*STRIDE+(k%KS)*DIL-PAD
    value=tl.load(X+((row//OL)*CI+k//KS)*L+loc,
                  (row<B*OL)&(loc>=0)&(loc<L),other=0.)
    tl.store(A+off,(rounded(value)*256.).to(tl.float16),row<B*OL)

@triton.jit
def gemm(A,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr,
         MODE:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    split=tl.program_id(2)
    blocks=tl.cdiv(K,BK*SPLIT)
    acc=tl.full((BM,BN),0.,tl.float32)
    for block in range(blocks):
        kk=(split*blocks+block)*BK+k
        lhs=tl.load(A+m[:,None]*K+kk[None,:],(m[:,None]<M)&(kk[None,:]<K),other=0.)
        rhs=tl.load(W+n[None,:]*K+kk[:,None],(n[None,:]<N)&(kk[:,None]<K),other=0.)
        if MODE==0:acc=tl.dot(lhs,rhs,acc,allow_tf32=False)
        else:acc=_rounding_boundary(acc+_rounding_boundary(tl.dot(lhs,rhs,allow_tf32=False)))
    tl.store(P+(split*M+m[:,None])*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def reduce(P,Bias,Y,M:tl.constexpr,N:tl.constexpr,OL:tl.constexpr,SPLIT:tl.constexpr,
           HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    split=tl.arange(0,SPLIT)
    values=tl.load(P+split[:,None]*(M*N)+off[None,:],off[None,:]<M*N,other=0.)
    result=tl.sum(values,axis=0)*(1./65536.)
    m=off//N;n=off%N
    if HAS_BIAS:result+=tl.load(Bias+n,off<M*N,other=0.)
    tl.store(Y+(m//OL*N+n)*OL+m%OL,result,off<M*N)

def workspace(x,half_weight,metadata,config):
    b,ci,length=x.shape;co,_,ks=half_weight.shape
    ol=(length+2*metadata['padding'][0]-metadata['dilation'][0]*(ks-1)-1)//metadata['stride'][0]+1
    values=[]
    for shape,dtype in (((b*ol,ci*ks),torch.float16),((config[5],b*ol,co),torch.float32)):
        size=1
        for dimension in shape:size*=dimension
        backing=torch.full((size+128,),1337.,device=x.device,dtype=dtype)
        values.append(backing[64:-64].reshape(shape))
    return tuple(values)

def verify_workspace(buffers):
    for value in buffers:
        assert torch.isfinite(value).all()
        assert (value._base[:64]==1337.).all() and (value._base[-64:]==1337.).all()

def launch(x,half_weight,bias,output,metadata,config,buffers):
    assert not torch.is_grad_enabled() and x.dtype==output.dtype==torch.float32
    assert x.is_contiguous() and output.is_contiguous() and half_weight.is_contiguous()
    assert half_weight.dtype==torch.float16 and metadata['groups']==1
    b,ci,length=x.shape;co,ci2,ks=half_weight.shape;assert ci==ci2
    stride,pad,dil=metadata['stride'][0],metadata['padding'][0],metadata['dilation'][0]
    ol=(length+2*pad-dil*(ks-1)-1)//stride+1
    bm,bn,bk,warps,stages,split,mode=config
    a,partial=buffers
    assert a.shape==(b*ol,ci*ks) and partial.shape==(split,b*ol,co)
    assert output.shape==(b,co,ol)
    im2col[(triton.cdiv(a.numel(),1024),)](x,a,b,ci,length,ks,ol,stride,pad,dil,1024,num_warps=4)
    result=gemm[(triton.cdiv(b*ol,bm),triton.cdiv(co,bn),split)](
        a,half_weight,partial,b*ol,co,ci*ks,bm,bn,bk,split,mode,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    reduce[(triton.cdiv(b*ol*co,256),)](partial,half_weight if bias is None else bias,output,
        b*ol,co,ol,split,bias is not None,256,num_warps=4,enable_fp_fusion=False)
    return result
