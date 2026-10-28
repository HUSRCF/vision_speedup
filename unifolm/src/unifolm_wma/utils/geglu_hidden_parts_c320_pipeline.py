"""Fresh Half hidden parts only; original FP32 residual output."""
import os
import torch
import triton
from .triton_geglu_hidden_parts_c320_consumer import _packed_three_product_gemm_residual
from .triton_geglu_hidden_parts_vector_store import _zero_geglu_hidden_parts_vector_store as _zero_geglu_hidden_parts
from .triton_geglu_subtile32 import _geglu_subtile32
from .geglu_output_allocation import empty as empty_geglu
from .gemm_output_allocation import empty as empty_gemm


def geglu_hidden_parts_c320(inputs,parts,count,bias=None):
    high,low=inputs
    if (torch.is_grad_enabled() or not high.is_cuda or high.ndim!=2
            or high.dtype!=torch.float16 or high.shape[1]!=320
            or any(p.shape!=high.shape or p.dtype!=torch.float16 or p.device!=high.device
                   or not p.is_contiguous() for p in inputs)):
        raise ValueError('Expected fresh C320 input parts')
    rows,inner=high.shape;channels=inner*4
    if len(parts)!=2 or any(p.shape!=(channels*2,inner) or p.dtype!=torch.float16
            or p.device!=high.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid original C320 GEGLU weight parts')
    if count.shape!=() or count.dtype!=torch.int64 or count.device!=high.device:
        raise ValueError('Invalid existing device predicate')
    if bias is not None and (bias.shape!=(channels*2,) or bias.dtype!=torch.float32
            or bias.device!=high.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 GEGLU bias')
    subtile=os.environ.get('UNIFOLM_WORLD_GEGLU_SUBTILE32','0')
    if subtile not in {'0','1'}:raise ValueError('Invalid world GEGLU K32 mode')
    use_subtile=(subtile=='1' and (rows,channels,inner)==(40960, 1280, 320)
                 and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
    kernel=_geglu_subtile32 if use_subtile else _zero_geglu_hidden_parts
    outputs=tuple(empty_geglu((rows,channels),device=high.device,dtype=torch.float16) for _ in range(2))
    kernel[(triton.cdiv(rows,64)*triton.cdiv(channels,64),)](
        *inputs,*parts,count,parts[0] if bias is None else bias,*outputs,outputs[0],outputs[0],
        rows,channels,inner,bias is not None,64,64,64,16,False,True,False,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return outputs


def packed_hidden_ffn320_residual(inputs,parts,residual,bias=None):
    high,low=inputs
    if (torch.is_grad_enabled() or not high.is_cuda or high.ndim!=2 or high.shape[1]!=1280
            or min(high.shape)<=0 or any(p.shape!=high.shape or p.device!=high.device
                or p.dtype!=torch.float16 or not p.is_contiguous() for p in inputs)):
        raise ValueError('Expected fresh C320 FFN hidden parts')
    rows,inner=high.shape;channels=parts[0].shape[0]
    if len(parts)!=2 or any(p.shape!=(channels,inner) or p.device!=high.device
            or p.dtype!=torch.float16 or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid FFN static parts')
    if (residual.shape!=(rows,channels) or residual.dtype!=torch.float32
            or residual.device!=high.device or not residual.is_contiguous()):
        raise ValueError('Invalid original FP32 residual')
    if bias is not None and (bias.shape!=(channels,) or bias.dtype!=torch.float32
            or bias.device!=high.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 final bias')
    output=empty_gemm('classic',(rows,channels),device=high.device,dtype=torch.float32)
    bm=64 if rows>=2560 else 32
    _packed_three_product_gemm_residual[(triton.cdiv(rows,bm)*triton.cdiv(channels,64),)](
        *inputs,*parts,parts[0] if bias is None else bias,residual,output,
        rows,channels,inner,bias is not None,bm,64,1.,256.,True,32,64,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return output
