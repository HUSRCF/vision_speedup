"""Numerical Conv1d prototype: explicit rounded operands, FP32 accumulation."""
import torch,triton
import triton.language as tl
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary

@triton.jit
def rounded(value):
    bits=value.to(tl.int32,bitcast=True)
    return ((bits+0x1000)&-8192).to(tl.float32,bitcast=True)

@triton.jit
def pack(W,H,SIZE:tl.constexpr,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(W+off,off<SIZE,other=0.)
    tl.store(H+off,(rounded(value)*256.).to(tl.float16),off<SIZE)

@triton.jit
def convolution(X,H,Bias,Y,B:tl.constexpr,CI:tl.constexpr,L:tl.constexpr,
                CO:tl.constexpr,KS:tl.constexpr,OL:tl.constexpr,
                STRIDE:tl.constexpr,PAD:tl.constexpr,DIL:tl.constexpr,
                HAS_BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,ACC_MODE:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    batch=row//OL;position=row%OL
    acc=tl.full((BM,BN),0.,tl.float32)
    compensation=tl.full((BM,BN),0.,tl.float32)
    for block in range(tl.cdiv(CI*KS,BK)):
        k=block*BK+kk
        channel=k//KS;tap=k%KS
        loc=position[:,None]*STRIDE+tap[None,:]*DIL-PAD
        mask=(row[:,None]<B*OL)&(k[None,:]<CI*KS)&(loc>=0)&(loc<L)
        value=tl.load(X+(batch[:,None]*CI+channel[None,:])*L+loc,mask,other=0.)
        x=(rounded(value)*256.).to(tl.float16)
        weight=tl.load(H+col[None,:]*(CI*KS)+k[:,None],
            (col[None,:]<CO)&(k[:,None]<CI*KS),other=0.)
        if ACC_MODE==0:
            acc=tl.dot(x,weight,acc,allow_tf32=False)
        else:
            block_value=_rounding_boundary(tl.dot(x,weight,allow_tf32=False))
            if ACC_MODE==1:
                acc=_rounding_boundary(acc+block_value)
            else:
                corrected=_rounding_boundary(block_value-compensation)
                total=_rounding_boundary(acc+corrected)
                compensation=_rounding_boundary(_rounding_boundary(total-acc)-corrected)
                acc=total
    result=acc*(1./65536.)
    if HAS_BIAS:result+=tl.load(Bias+col,col<CO,other=0.)[None,:]
    offset=(batch[:,None]*CO+col[None,:])*OL+position[:,None]
    tl.store(Y+offset,result,(row[:,None]<B*OL)&(col[None,:]<CO))

def launch(x,half_weight,bias,output,metadata,config):
    assert not torch.is_grad_enabled() and x.dtype==torch.float32 and x.is_contiguous()
    assert half_weight.dtype==torch.float16 and half_weight.is_contiguous()
    assert output.dtype==torch.float32 and output.is_contiguous() and metadata['groups']==1
    batch,ci,length=x.shape;co,ci2,ks=half_weight.shape;assert ci==ci2
    stride,pad,dilation=metadata['stride'][0],metadata['padding'][0],metadata['dilation'][0]
    ol=(length+2*pad-dilation*(ks-1)-1)//stride+1
    assert output.shape==(batch,co,ol)
    bm,bn,bk,warps,stages=config[:5]
    acc_mode=config[5] if len(config)>5 else 0
    return convolution[(triton.cdiv(batch*ol,bm),triton.cdiv(co,bn))](
        x,half_weight,half_weight if bias is None else bias,output,batch,ci,length,co,ks,ol,
        stride,pad,dilation,bias is not None,bm,bn,bk,acc_mode,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
