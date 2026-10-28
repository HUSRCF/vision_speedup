"""Merge accepted spatial QKV projections and K/V decomposition in world only."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_zero_weight_low_linear import weight_low_count
from .triton_qkv_packed_attention import qkv_packed_attention


SHAPES={(16,2560,320),(16,640,640)}


def compatible(module):
    projections=module.to_q,module.to_k,module.to_v
    channels=projections[0].in_features
    return (channels in {320,640} and module.heads==channels//64 and module.dim_head==64
            and module.scale==.125 and not module.relative_position
            and getattr(module,'_compensated_spatial_attention',False)
            and getattr(module,'_world_zero_weight_low_qkv_installed',False)
            and getattr(module,'_compensated_self_qkv_enabled',False)
            and module._inference_attention_layout=='bmhk'
            and module._inference_attention_dtype==torch.float32
            and module._inference_projection_dtype==torch.float32
            and all(isinstance(p,torch.nn.Linear) and p.bias is None
                    and tuple(p.weight.shape)==(channels,channels) for p in projections))


def make_forward(module):
    original=module.forward

    def forward(value,context=None,mask=None):
        projections=module.to_q,module.to_k,module.to_v
        if (not compatible(module) or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or context is not None or mask is not None
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype!=torch.float32 or not value.is_contiguous()
                or tuple(value.shape) not in SHAPES
                or any(p.training or p.weight.dtype!=torch.float32 or p.weight.device!=value.device
                       or not p.weight.is_contiguous() for p in projections)
                or any(m.training for m in module.to_out.modules())):
            return original(value,context=context,mask=mask)
        weights=tuple(p.weight for p in projections)
        try:
            key=tuple((w.data_ptr(),w._version,tuple(w.shape),w.dtype,w.device) for w in weights)
        except RuntimeError:
            return original(value,context=context,mask=mask)
        channels=value.shape[-1]
        # Share exactly the existing static QKV cache and update contract.
        cache=getattr(module,'_world_zero_weight_low_qkv_cache',None)
        if cache is None or cache[0]!=key:
            if torch.cuda.is_current_stream_capturing():
                return original(value,context=context,mask=mask)
            reuse=(cache is not None and cache[1].shape==(3*channels,channels)
                   and cache[1].device==value.device)
            if reuse:
                combined=cache[1];torch.cat(weights,dim=0,out=combined)
                parts=pack_half_weight(combined,out=cache[2])
            else:
                combined=torch.cat(weights,dim=0);parts=pack_half_weight(combined)
            count=weight_low_count(parts,out=cache[3] if reuse else None)
            cache=key,combined,parts,count
            module._world_zero_weight_low_qkv_cache=cache
        return module.to_out(qkv_packed_attention(value,cache[2],cache[3],scale=module.scale))

    return forward


def install_world_qkv_packed_attention(diffusion,scope='all'):
    from unifolm_wma.modules.attention import SpatialTransformer
    if scope not in {'all','c320','c640'}:raise ValueError('Invalid QKV attention fusion scope')
    selected=[];seen=set()
    for parent in diffusion.modules():
        if not isinstance(parent,SpatialTransformer):continue
        for block in parent.transformer_blocks:
            module=block.attn1
            channels=module.to_q.in_features
            if channels not in {320,640} or scope not in {'all','c'+str(channels)}:continue
            if id(module) in seen or not compatible(module) or getattr(module,'_qkv_packed_attention_installed',False):
                raise ValueError('Expected previously accepted spatial QKV/attention configuration')
            seen.add(id(module));selected.append(module)
    for module in selected:
        module.forward=make_forward(module)
        module._qkv_packed_attention_installed=True
    return len(selected)
