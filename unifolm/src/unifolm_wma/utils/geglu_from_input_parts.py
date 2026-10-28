"""Launch the original world GEGLU kernel with explicit fresh input parts."""
import torch
import triton
from .triton_zero_weight_low_geglu import _zero_geglu
from .geglu_output_allocation import empty as empty_geglu


def geglu_from_input_parts(inputs,parts,count,bias=None,*,shape=None):
    high,low=inputs
    if (torch.is_grad_enabled() or not high.is_cuda or high.dtype!=torch.float16
            or high.ndim!=2 or low.shape!=high.shape or low.dtype!=high.dtype
            or low.device!=high.device or not high.is_contiguous() or not low.is_contiguous()
            or high.shape[1] not in {320,640,1280} or min(high.shape)<=0):
        raise ValueError('Expected two explicitly refreshed world GEGLU input parts')
    rows,inner=high.shape
    if (len(parts)!=2 or parts[0].shape[0]%2 or any(p.ndim!=2 or p.shape!=parts[0].shape
            or p.shape[1]!=inner or p.dtype!=torch.float16 or p.device!=high.device
            or not p.is_contiguous() for p in parts)):
        raise ValueError('Invalid GEGLU static weight parts')
    channels=parts[0].shape[0]//2
    if count.shape!=() or count.dtype!=torch.int64 or count.device!=high.device:
        raise ValueError('Invalid persistent weight residual predicate')
    if bias is not None and (bias.shape!=(2*channels,) or bias.dtype!=torch.float32
            or bias.device!=high.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 GEGLU bias')
    shape=(rows,channels) if shape is None else tuple(shape)
    if shape[-1]!=channels or torch.Size(shape).numel()!=rows*channels:
        raise ValueError('Invalid original public output shape')
    output=empty_geglu(shape,device=high.device,dtype=torch.float32)
    bn=64 if inner in {320,640} else 32
    group=16 if inner==320 else 32
    _zero_geglu[(triton.cdiv(rows,64)*triton.cdiv(channels,bn),)](
        *inputs,*parts,count,parts[0] if bias is None else bias,output,output,output,
        rows,channels,inner,bias is not None,64,bn,64,group,False,inner in {320,640},False,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return output
