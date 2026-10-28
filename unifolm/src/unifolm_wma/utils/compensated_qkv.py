"""One input split and compensated GEMM, with three separate FP32 Q/K/V views."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_prepacked_input_linear import prepacked_input_linear


def fused_compensated_qkv(module, value):
    if getattr(module, '_initial_bk32_qkv_installed', False):
        from .initial_bk32_qkv import try_initial_bk32_qkv
        result = try_initial_bk32_qkv(module, value)
        if result is not None:
            return result
    if getattr(module, '_world_zero_weight_low_qkv_installed', False):
        from .world_zero_weight_low_qkv import try_world_zero_weight_low_qkv
        result = try_world_zero_weight_low_qkv(module, value)
        if result is not None:
            return result
    if getattr(module,'_decision_primary_grouped_qkv_installed',False):
        from .decision_primary_grouped_qkv import try_primary_grouped_qkv
        result=try_primary_grouped_qkv(module,value)
        if result is not None:return result
    projections = module.to_q, module.to_k, module.to_v
    def original():
        return tuple(projection(value) for projection in projections)
    weights = tuple(projection.weight for projection in projections)
    if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
            or not value.is_cuda or value.dtype != torch.float32
            or not value.is_contiguous() or value.ndim < 2
            or any(projection.training or projection.bias is not None
                   for projection in projections)
            or any(weight.dtype != torch.float32 or weight.device != value.device
                   or not weight.is_contiguous() or weight.shape != weights[0].shape
                   for weight in weights)
            or min(weights[0].shape) < 640
            or value.shape[-1] != weights[0].shape[1]
            or value.numel() * weights[0].shape[0] < 500_000_000):
        return original()
    key = tuple((w.data_ptr(), w._version, tuple(w.shape), w.dtype, w.device)
                for w in weights)
    cached = getattr(module, '_compensated_qkv_weights', None)
    if cached is None or cached[0] != key:
        if torch.cuda.is_current_stream_capturing():
            # No unrefreshed weight copies may become constants in a cold graph.
            return original()
        combined = torch.cat(weights, dim=0)
        if cached is not None and cached[1].shape == combined.shape and cached[1].device == combined.device:
            cached[1].copy_(combined)
            combined = cached[1]
            parts = pack_half_weight(combined, out=cached[2])
        else:
            parts = pack_half_weight(combined)
        cached = (key, combined, parts)
        module._compensated_qkv_weights = cached
    rows = value.numel() // value.shape[-1]
    packed = prepacked_input_linear(value, cached[1], weight_parts=cached[2],
        block_m=64 if rows >= 2560 else 32, block_k=64, group_m=32,
        num_stages=2, output_groups=3)
    return tuple(packed.unbind(0))


def install_compensated_qkv(model):
    from unifolm_wma.modules.attention import CrossAttention
    count = 0
    for module in model.modules():
        if not isinstance(module, CrossAttention):
            continue
        projections = module.to_q, module.to_k, module.to_v
        if (all(isinstance(p, torch.nn.Linear) and p.bias is None for p in projections)
                and min(projections[0].weight.shape) >= 640
                and all(p.weight.shape == projections[0].weight.shape for p in projections)):
            if module._fused_self_qkv_enabled:
                raise ValueError('Compensated QKV cannot combine with native fused QKV')
            module._compensated_self_qkv_enabled = True
            count += 1
    return count
