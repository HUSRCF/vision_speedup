"""Fuse only initial512 world self QKV with the existing BK32 arithmetic."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_zero_weight_low_linear import weight_low_count
from .triton_zero_weight_low_linear_bk32 import zero_weight_low_linear_bk32


def try_initial_bk32_qkv(module, value):
    projections = module.to_q, module.to_k, module.to_v
    if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
            or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
            or not value.is_cuda or value.dtype != torch.float32 or value.ndim < 2
            or not value.is_contiguous() or value.shape[-1] != 512
            or value.numel() // 512 != 40960
            or any(p.training or p.bias is not None or p.weight.dtype != torch.float32
                   or p.weight.device != value.device or not p.weight.is_contiguous()
                   or tuple(p.weight.shape) != (512, 512) for p in projections)):
        return None
    weights = tuple(p.weight for p in projections)
    try:
        key = tuple((w.data_ptr(), w._version, tuple(w.shape), w.dtype, w.device) for w in weights)
    except RuntimeError:
        return None
    cache = getattr(module, '_initial_bk32_qkv_cache', None)
    if cache is None or cache[0] != key:
        if torch.cuda.is_current_stream_capturing():
            return None
        reuse = (cache is not None and cache[1].shape == (1536, 512)
                 and cache[1].device == value.device)
        if reuse:
            combined = cache[1]
            torch.cat(weights, dim=0, out=combined)
            parts = pack_half_weight(combined, out=cache[2])
        else:
            combined = torch.cat(weights, dim=0)
            parts = pack_half_weight(combined)
        count = weight_low_count(parts, out=cache[3] if reuse else None)
        cache = key, combined, parts, count
        module._initial_bk32_qkv_cache = cache
    result = zero_weight_low_linear_bk32(value, cache[2], cache[3], output_groups=3)
    return tuple(result.unbind(0))


def install_initial_bk32_qkv(diffusion):
    from unifolm_wma.modules.attention import CrossAttention
    selected = []
    for name, module in diffusion.named_modules():
        if not name.startswith('init_attn.') or not isinstance(module, CrossAttention):
            continue
        projections = module.to_q, module.to_k, module.to_v
        if any(not isinstance(p, torch.nn.Linear) or p.bias is not None
               or tuple(p.weight.shape) != (512, 512) for p in projections):
            continue
        if (module._fused_self_qkv_enabled
                or getattr(module, '_compensated_self_qkv_enabled', False)
                or getattr(module, '_initial_bk32_qkv_installed', False)
                or any(not getattr(p, '_initial_world_linear_installed', False)
                       or not getattr(p, '_initial_decision_linear_installed', False)
                       for p in projections)):
            raise ValueError('Expected accepted unmerged initial512 projection wrappers')
        selected.append(module)
    for module in selected:
        module._initial_bk32_qkv_installed = True
        module._compensated_self_qkv_enabled = True
    return len(selected)
