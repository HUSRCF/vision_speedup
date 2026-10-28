"""C320 host interface to the unchanged accepted device-predicated shader."""
import torch
import triton
from .ffn1280_zero_residual import _zero_low_gemm_residual, shared_final_low_count
from .gemm_output_allocation import empty as empty_gemm


def zero_ffn320_hidden_residual(inputs,parts,count,residual,bias=None):
    high,low=inputs
    if (torch.is_grad_enabled() or not high.is_cuda or high.ndim!=2 or high.shape[1]!=1280
            or min(high.shape)<=0 or any(p.shape!=high.shape or p.device!=high.device
                or p.dtype!=torch.float16 or not p.is_contiguous() for p in inputs)):
        raise ValueError('Expected fresh C320 hidden Half parts')
    rows,inner=high.shape;channels=parts[0].shape[0]
    if len(parts)!=2 or any(p.shape!=(channels,inner) or p.device!=high.device
            or p.dtype!=torch.float16 or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid original C320 FFN static parts')
    if count.shape!=() or count.dtype!=torch.int64 or count.device!=high.device:
        raise ValueError('Invalid device C320 FFN residual count')
    if (residual.shape!=(rows,channels) or residual.dtype!=torch.float32
            or residual.device!=high.device or not residual.is_contiguous()):
        raise ValueError('Invalid FP32 residual')
    if bias is not None and (bias.shape!=(channels,) or bias.dtype!=torch.float32
            or bias.device!=high.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    output=empty_gemm('classic',(rows,channels),device=high.device,dtype=torch.float32)
    bm=64 if rows>=2560 else 32
    _zero_low_gemm_residual[(triton.cdiv(rows,bm)*triton.cdiv(channels,64),)](
        *inputs,*parts,count,parts[0] if bias is None else bias,residual,output,
        rows,channels,inner,bias is not None,bm,32,1,
        num_warps=4,num_stages=3,enable_fp_fusion=False)
    return output
