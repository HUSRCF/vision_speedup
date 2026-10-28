"""World-only QKV fusion with a graph-visible weight residual predicate."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_zero_weight_low_linear import zero_weight_low_linear, weight_low_count


def try_world_zero_weight_low_qkv(module, value):
    projections = module.to_q, module.to_k, module.to_v
    channels = projections[0].in_features
    selected = {(40960, 320), (10240, 640), (2560, 1280), (640, 1280)}
    if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
            or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
            or not value.is_cuda or value.dtype != torch.float32 or value.ndim < 2
            or not value.is_contiguous() or value.shape[-1] != channels
            or (value.numel() // channels, channels) not in selected
            or any(p.training or p.bias is not None or p.weight.dtype != torch.float32
                   or p.weight.device != value.device or not p.weight.is_contiguous()
                   or tuple(p.weight.shape) != (channels, channels) for p in projections)):
        return None
    weights = tuple(p.weight for p in projections)
    try:
        key = tuple((w.data_ptr(), w._version, tuple(w.shape), w.dtype, w.device) for w in weights)
    except RuntimeError:
        return None
    cache = getattr(module, '_world_zero_weight_low_qkv_cache', None)
    if cache is None or cache[0] != key:
        if torch.cuda.is_current_stream_capturing():
            return None
        reuse = (cache is not None and cache[1].shape == (3 * channels, channels)
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
        module._world_zero_weight_low_qkv_cache = cache
    result = zero_weight_low_linear(value, cache[2], cache[3], output_groups=3)
    return tuple(result.unbind(0))


def install_world_zero_weight_low_qkv(diffusion, scope='all'):
    from unifolm_wma.modules.attention import CrossAttention
    if scope not in {'all', '320', 'wide'}:
        raise ValueError('Invalid world zero-weight-residual QKV scope')
    selected = []
    for module in diffusion.modules():
        if not isinstance(module, CrossAttention):
            continue
        projections = module.to_q, module.to_k, module.to_v
        channels = projections[0].in_features
        if (channels not in {320, 640, 1280}
                or (scope == '320' and channels != 320)
                or (scope == 'wide' and channels == 320)
                or any(not isinstance(p, torch.nn.Linear) or p.bias is not None
                       or tuple(p.weight.shape) != (channels, channels) for p in projections)):
            continue
        if (not getattr(module, '_compensated_self_qkv_enabled', False)
                or (channels == 320 and not getattr(module, '_decision_primary_grouped_qkv_installed', False))
                or getattr(module, '_world_zero_weight_low_qkv_installed', False)):
            raise ValueError('Expected accepted grouped QKV stage configuration')
        selected.append(module)
    for module in selected:
        module._world_zero_weight_low_qkv_installed = True
    return len(selected)
