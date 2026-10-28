"""Residual add in the original compensated FFN GEMM epilogue."""
import torch
import triton
from .triton_ffn640_residual_epilogue_kernel import _two_product_gemm_residual
from .gemm_output_allocation import empty as empty_gemm


def ffn640_residual_epilogue(value,parts,residual,bias=None):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim<1 or not value.is_contiguous() or len(parts)!=2
            or parts[0].ndim!=2 or parts[0].shape[1]!=value.shape[-1]
            or any(p.shape!=parts[0].shape or p.device!=value.device or p.dtype!=torch.float16
                   or not p.is_contiguous() for p in parts)):
        raise ValueError('Expected original GEMM FP32 input and static Half weight parts')
    channels,inner=parts[0].shape;rows=value.numel()//inner
    shape=(*value.shape[:-1],channels)
    if min(rows,channels,inner)<=0 or residual.shape!=shape or residual.dtype!=torch.float32 or residual.device!=value.device or not residual.is_contiguous():
        raise ValueError('Invalid original FP32 residual output shape')
    if bias is not None and (bias.shape!=(channels,) or bias.dtype!=torch.float32 or bias.device!=value.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 GEMM bias')
    bm=64
    output=empty_gemm('classic',shape,device=value.device,dtype=torch.float32)
    _two_product_gemm_residual[(triton.cdiv(rows,bm)*triton.cdiv(channels,128),)](
        value,*parts,parts[0] if bias is None else bias,residual,output,
        rows,channels,inner,bias is not None,True,bm,128,64,32,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return output
