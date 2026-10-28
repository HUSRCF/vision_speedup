"""FP32 temporal attention, three-product Tensor Core QK/PV, native softmax tree."""
import torch,triton
import triton.language as tl
from unifolm_wma.utils.triton_compensated_attention import _three_product
from unifolm_wma.utils.triton_half_compensated_linear import _half_split

@triton.jit
def attention(Q,K,V,O,QK,P,HEADS:tl.constexpr,SCALE:tl.constexpr,DEBUG:tl.constexpr):
    batch_head=tl.program_id(0);batch=batch_head//HEADS;head=batch_head%HEADS
    frame=tl.arange(0,16);channel=tl.arange(0,64)
    origin=batch*16*HEADS*64+head*64
    q=tl.load(Q+origin+frame[:,None]*HEADS*64+channel[None,:])
    k=tl.load(K+origin+frame[None,:]*HEADS*64+channel[:,None])
    value=tl.load(V+origin+frame[:,None]*HEADS*64+channel[None,:])
    aq,ak,av=tl.abs(q),tl.abs(k),tl.abs(value)
    largest=tl.maximum(tl.maximum(tl.max(tl.max(aq,0),0),tl.max(tl.max(ak,0),0)),tl.max(tl.max(av,0),0))
    smallest=tl.minimum(tl.minimum(tl.min(tl.min(tl.where(aq>0.,aq,float('inf')),0),0),tl.min(tl.min(tl.where(ak>0.,ak,float('inf')),0),0)),tl.min(tl.min(tl.where(av>0.,av,float('inf')),0),0))
    use_native=(largest>=128.)|(smallest<0.00000095367431640625)
    if use_native:
        raw=tl.dot(q,k,allow_tf32=False)
    else:
        qh,ql=_half_split(q,256.);kh,kl=_half_split(k,256.)
        raw=_three_product(qh,ql,kh,kl,1./65536.)
    off=batch_head*256+frame[:,None]*16+frame[None,:]
    if DEBUG:tl.store(QK+off,raw)
    scores=raw*SCALE
    probabilities=tl.math.exp(scores-tl.max(scores,1)[:,None])
    total=tl.sum(tl.reshape(probabilities,(16,2,8)),1)
    total=tl.sum(tl.reshape(total,(16,2,4)),1)
    total=tl.sum(tl.reshape(total,(16,2,2)),1)
    denominator=tl.sum(total,1)[:,None]
    probabilities=tl.math.div_rn(probabilities,denominator)
    if DEBUG:tl.store(P+off,probabilities)
    if use_native:
        out=tl.dot(probabilities,value,allow_tf32=False)
    else:
        ph,pl=_half_split(probabilities,256.);vh,vl=_half_split(value,256.)
        out=_three_product(ph,pl,vh,vl,1./65536.)
    tl.store(O+origin+frame[:,None]*HEADS*64+channel[None,:],out)

def launch(q,k,v,out,qk,probability,heads,scale,warps,debug):
    assert not torch.is_grad_enabled() and not torch.backends.cuda.matmul.allow_tf32
    assert q.shape==k.shape==v.shape==out.shape and q.ndim==3 and q.shape[1:]==(16,heads*64)
    assert all(t.is_cuda and t.dtype==torch.float32 and t.is_contiguous() and t.device==q.device for t in (q,k,v,out))
    return attention[(q.shape[0]*heads,)](q,k,v,out,qk,probability,heads,scale,debug,
        num_warps=warps,num_stages=1,enable_fp_fusion=False)
