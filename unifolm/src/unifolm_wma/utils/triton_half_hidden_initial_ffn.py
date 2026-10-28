"""Materialize the final primary GEMM's existing FP16 input directly.

The projection, bias and native activation still run in their accepted arithmetic.
Only the internal activation storage changes: the next GEMM already casts it to
FP16. Public FFN output and both master weights remain FP32 in a runtime wrapper.
"""
import torch
import triton
from .triton_inline_primary_geglu import _inline_primary_geglu
from .triton_primary_narrow_linear import _primary


def half_hidden_ffn(value, first_high, second_high, first_bias=None, second_bias=None,
                    *, debug_hidden=False, debug_poison=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim<1 or value.shape[-1]==0 or not value.is_contiguous()):
        raise ValueError('Expected contiguous FP32 CUDA inference input')
    weights = (first_high,second_high)
    if any(w.ndim!=2 or w.dtype!=torch.float16 or w.device!=value.device
           or not w.is_contiguous() for w in weights):
        raise ValueError('Expected cached scaled FP16 weights')
    inner = value.shape[-1]
    rows = value.numel()//inner
    hidden, outputs = first_high.shape[0]//2,second_high.shape[0]
    if (first_high.shape[0]%2 or first_high.shape[1]!=inner
            or second_high.shape[1]!=hidden or min(rows,inner,hidden,outputs)<=0):
        raise ValueError('Invalid FFN dimensions')
    for bias,width in [(first_bias,2*hidden),(second_bias,outputs)]:
        if bias is not None and (bias.shape!=(width,) or bias.dtype!=torch.float32
                or bias.device!=value.device or not bias.is_contiguous()):
            raise ValueError('Invalid FP32 FFN bias')
    middle = torch.empty((*value.shape[:-1],hidden),device=value.device,dtype=torch.float16)
    output = torch.empty((*value.shape[:-1],outputs),device=value.device,dtype=torch.float32)
    if debug_poison:
        middle.fill_(float('nan'))
        output.fill_(float('nan'))
    _inline_primary_geglu[(triton.cdiv(rows,64)*triton.cdiv(hidden,128),)](
        value,first_high,first_high if first_bias is None else first_bias,
        middle,middle,middle,rows,hidden,inner,first_bias is not None,
        64,128,32,False,True,num_warps=4,num_stages=2,enable_fp_fusion=False)
    _primary[(triton.cdiv(rows,64)*triton.cdiv(outputs,128),)](
        middle,second_high,second_high if second_bias is None else second_bias,output,
        rows,outputs,hidden,second_bias is not None,64,128,32,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return (output,middle) if debug_hidden else output
