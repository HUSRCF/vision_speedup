"""World norm2 packing into the original QKV and FP32 attention kernels."""
import torch
import triton
from .residual_norm_half_pack import residual_norm_half_pack
from .triton_zero_weight_low_linear import _zero_low_gemm
from .triton_temporal_attention_fp32 import fused_temporal_attention_fp32
from .gemm_output_allocation import empty

def cache_key(module):
    return tuple((p.weight.data_ptr(),p.weight._version,tuple(p.weight.shape),p.weight.dtype,p.weight.device)
                 for p in [module.to_q,module.to_k,module.to_v])

def try_norm2_qkv_pair(block,left,right,context,mask):
    module=block.attn2;norm=block.norm2;c=right.shape[-1]
    if (not getattr(block,'_world_norm2_qkv_pack_enabled',False) or context is not None or mask is not None
            or torch.is_grad_enabled() or torch.is_autocast_enabled()
            or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
            or any(m.training for m in [block,module,norm,*module.modules()])
            or not getattr(module,'_world_zero_weight_low_qkv_installed',False)
            or not module._inference_fused_temporal_attention_fp32 or module.relative_position
            or module.temporal_length!=16 or module.dim_head!=64
            or right.ndim!=3 or right.shape[1]!=16
            or (right.numel()//c,c) not in {(40960,320),(10240,640),(2560,1280),(640,1280)}
            or any(not p.is_cuda or p.dtype!=torch.float32 or p.device!=right.device
                   or not p.is_contiguous() for p in [left,right,norm.weight,norm.bias])
            or left.shape!=right.shape):return None
    projections=[module.to_q,module.to_k,module.to_v]
    if any(p.training or p.bias is not None or p.weight.dtype!=torch.float32
           or p.weight.device!=right.device or not p.weight.is_contiguous()
           or tuple(p.weight.shape)!=(c,c) for p in projections):return None
    try:key=cache_key(module)
    except RuntimeError:return None
    cache=getattr(module,'_world_zero_weight_low_qkv_cache',None)
    # The original warm/cold/stale path owns cache creation and refresh.
    if cache is None or cache[0]!=key:return None
    residual,*inputs=residual_norm_half_pack(left,right,norm)
    rows=right.numel()//c;outputs=3*c;large=rows>=2560
    packed=empty('classic',(3,*right.shape),device=right.device,dtype=torch.float32)
    _zero_low_gemm[(triton.cdiv(rows,64 if large else 32)*triton.cdiv(outputs,64),)](
        *inputs,*cache[2],cache[3],cache[2][0],packed,rows,outputs,c,False,
        64 if large else 32,1 if large else 32,3,num_warps=4,
        num_stages=3 if large else 2,enable_fp_fusion=False)
    q,k,v=packed.unbind(0)
    attended=module.to_out(fused_temporal_attention_fp32(q,k,v,module.heads,module.scale))
    return residual,attended


def requested():
    import os
    mode=os.environ.get('UNIFOLM_WORLD_NORM2_QKV_PACK','0')
    if mode not in {'0','1'}:raise ValueError('Invalid norm2 QKV packing mode')
    if mode=='0':return False
    for key,value in {'UNIFOLM_FUSED_RESIDUAL_NORM': '1', 'UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV': 'all', 'UNIFOLM_WORLD_ZERO_QKV_STAGES': '3-large', 'UNIFOLM_WORLD_ZERO_QKV_GROUP': '1-large', 'UNIFOLM_NORM_BUFFER_NOFILL': 'all', 'UNIFOLM_NATIVE_SGEMM_FP16': '1', 'UNIFOLM_WORLD_NORM_PRESPLIT_TEMPORAL': '1'}.items():
        if os.environ.get(key)!=value:raise ValueError('Norm2 QKV packing requires '+key+'='+value)
    return True

def install(root):
    from unifolm_wma.modules.attention import BasicTransformerBlock
    selected=[b for b in root.modules() if isinstance(b,BasicTransformerBlock)
              and b.attn2.temporal_length==16 and b.norm2.normalized_shape[0] in {320,640,1280}]
    if len(selected)!=16 or any(getattr(b,'_world_norm2_qkv_pack_enabled',False) for b in selected):
        raise ValueError('Expected exactly16 unmodified norm2 temporal blocks')
    for block in selected:block._world_norm2_qkv_pack_enabled=True
    return len(selected)
