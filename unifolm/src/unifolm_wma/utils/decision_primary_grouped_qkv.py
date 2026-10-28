"""Stage-guarded C320 QKV fusion with current primary GEMM arithmetic."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_primary_grouped_qkv import primary_grouped_qkv


def try_primary_grouped_qkv(module,value):
    projections=(module.to_q,module.to_k,module.to_v)
    if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
            or not torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
            or not value.is_cuda or value.dtype!=torch.float32 or value.ndim<2
            or value.shape[-1]!=320 or value.numel()//320!=40960 or not value.is_contiguous()
            or any(p.training or p.bias is not None or tuple(p.weight.shape)!=(320,320)
                or p.weight.dtype!=torch.float32 or p.weight.device!=value.device
                or not p.weight.is_contiguous() for p in projections)):
        return None
    weights=tuple(p.weight for p in projections)
    try:key=tuple((w.data_ptr(),w._version,w.dtype,w.device,tuple(w.shape)) for w in weights)
    except RuntimeError:return None
    cache=getattr(module,'_primary_grouped_qkv_cache',None)
    if cache is None or cache[0]!=key:
        if torch.cuda.is_current_stream_capturing():return None
        reuse=cache is not None and cache[1].shape==(960,320) and cache[1].device==value.device
        if reuse:
            combined=cache[1]
            torch.cat(weights,dim=0,out=combined)
            parts=pack_half_weight(combined,out=cache[2])
        else:
            combined=torch.cat(weights,dim=0)
            parts=pack_half_weight(combined)
        cache=(key,combined,parts)
        module._primary_grouped_qkv_cache=cache
    packed=primary_grouped_qkv(value,cache[2][0],tile=(128,128,32,4,3))
    return tuple(packed.unbind(0))


def install_decision_primary_grouped_qkv(diffusion,scope='temporal'):
    from unifolm_wma.modules.attention import CrossAttention
    if scope not in {'all','temporal'}:raise ValueError('Invalid grouped QKV scope')
    count=0
    for module in diffusion.modules():
        if not isinstance(module,CrossAttention):continue
        if scope=='temporal' and module.temporal_length is None:continue
        projections=(module.to_q,module.to_k,module.to_v)
        if any(not isinstance(p,torch.nn.Linear) or p.bias is not None
                or tuple(p.weight.shape)!=(320,320) for p in projections):continue
        if (module._fused_self_qkv_enabled
                or getattr(module,'_compensated_self_qkv_enabled',False)
                or getattr(module,'_decision_primary_grouped_qkv_installed',False)
                or any(not getattr(p,'_decision_primary_narrow_installed',False) for p in projections)):
            raise ValueError('Grouped QKV requires unmerged accepted primary C320 projections')
        module._compensated_self_qkv_enabled=True
        module._compensated_qkv_narrow_enabled=False
        module._decision_primary_grouped_qkv_installed=True
        count+=1
    return count
