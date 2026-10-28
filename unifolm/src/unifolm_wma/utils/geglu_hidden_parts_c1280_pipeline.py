"""Fresh Half hidden parts only; original FP32 residual output."""
import os
import torch
import triton
from .triton_geglu_hidden_parts_c1280_consumer import _prepacked_gemm_residual
from .triton_geglu_hidden_parts_vector_store import _zero_geglu_hidden_parts_vector_store as _zero_geglu_hidden_parts
from .triton_geglu_both_subtile32 import _both_geglu_subtile32
from .geglu_output_allocation import empty as empty_geglu
from .gemm_output_allocation import empty as empty_gemm


def geglu_hidden_parts_c1280(inputs,parts,count,bias=None):
    high,low=inputs
    if (torch.is_grad_enabled() or not high.is_cuda or high.ndim!=2
            or high.dtype!=torch.float16 or high.shape[1]!=1280
            or any(p.shape!=high.shape or p.dtype!=torch.float16 or p.device!=high.device
                   or not p.is_contiguous() for p in inputs)):
        raise ValueError('Expected fresh C1280 input parts')
    rows,inner=high.shape;channels=inner*4
    if len(parts)!=2 or any(p.shape!=(channels*2,inner) or p.dtype!=torch.float16
            or p.device!=high.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid original C1280 GEGLU weight parts')
    if count.shape!=() or count.dtype!=torch.int64 or count.device!=high.device:
        raise ValueError('Invalid existing device predicate')
    if bias is not None and (bias.shape!=(channels*2,) or bias.dtype!=torch.float32
            or bias.device!=high.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 GEGLU bias')
    subtile_mode=os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32','0')
    if subtile_mode not in {'0','1'}:
        raise ValueError('Invalid C1280 GEGLU subtile mode')
    subtile_selected=(subtile_mode=='1' and rows in {2560,640}
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cuda.matmul.allow_tf32)
    tiles_mode=os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32_TILES','0')
    if tiles_mode not in {'0','1'}:
        raise ValueError('Invalid C1280 GEGLU subtile tile mode')
    tuned_tiles=tiles_mode=='1' and subtile_selected
    bm=(64 if rows==2560 else 32) if tuned_tiles else 64
    bn=64 if tuned_tiles else 32
    group=16 if tuned_tiles else 32
    kernel=_both_geglu_subtile32 if subtile_selected else _zero_geglu_hidden_parts
    outputs=tuple(empty_geglu((rows,channels),device=high.device,dtype=torch.float16) for _ in range(2))
    kernel[(triton.cdiv(rows,bm)*triton.cdiv(channels,bn),)](
        *inputs,*parts,count,parts[0] if bias is None else bias,*outputs,outputs[0],outputs[0],
        rows,channels,inner,bias is not None,bm,bn,64,group,False,False,False,
        num_warps=4,num_stages=2 if subtile_selected else int(os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_STAGES','2')),enable_fp_fusion=False)
    return outputs


def packed_hidden_ffn1280_residual(inputs,parts,residual,bias=None):
    high,low=inputs
    if (torch.is_grad_enabled() or not high.is_cuda or high.ndim!=2 or high.shape[1]!=5120
            or min(high.shape)<=0 or any(p.shape!=high.shape or p.device!=high.device
                or p.dtype!=torch.float16 or not p.is_contiguous() for p in inputs)):
        raise ValueError('Expected fresh C1280 FFN hidden parts')
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
    _prepacked_gemm_residual[(triton.cdiv(rows,bm)*triton.cdiv(channels,64),)](
        *inputs,*parts,parts[0] if bias is None else bias,residual,output,
        rows,channels,inner,bias is not None,bm,64,64,32,1,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return output
