import torch
import triton
from .triton_half_compensated_linear import pack_half_weight
from .triton_qkv_packed_attention import _qkv_pack
from .gemm_output_allocation import empty as empty_gemm
from .half_pack_allocation import empty_like as empty_half
from .attention_hidden_parts_pipeline import attention_hidden_parts,packed_attention_output

def qkv_attention_hidden_projection(value,parts,count,output_parts,output_bias=None,*,scale=.125):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim!=3 or not value.is_contiguous() or value.shape[-1] not in {320,640}
            or min(value.shape)<=0):
        raise ValueError('Expected contiguous CUDA FP32 BMC inference C320/640')
    batch,length,channels=value.shape
    if len(parts)!=2 or any(p.shape!=(channels*3,channels) or p.dtype!=torch.float16
            or p.device!=value.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid accepted packed QKV master parts')
    if count.shape!=() or count.dtype!=torch.int64 or count.device!=value.device:
        raise ValueError('Invalid graph-visible weight residual predicate')
    if scale!=.125:raise ValueError('Expected accepted H64 spatial attention scale')
    rows=batch*length
    packed=pack_half_weight(value.view(rows,channels),weight_scale=1.)
    query=empty_gemm('classic',value.shape,device=value.device,dtype=torch.float32)
    kh,kl,vh,vl=(empty_half(query) for _ in range(4))
    bm=64 if rows>=2560 else 32
    _qkv_pack[(triton.cdiv(rows,bm)*triton.cdiv(channels*3,64),)](
        *packed,*parts,count,query,kh,kl,vh,vl,rows,channels,bm,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    heads=channels//64
    shaped=query.view(batch,length,heads,64)
    hidden=attention_hidden_parts(shaped,(kh,kl,vh,vl),mode='vector')
    return packed_attention_output(hidden,output_parts,output_bias)
