"""Experimental autocast backbone with original FP32 weights and residuals."""
import torch


def _active(module, value):
    return (not module.training and not torch.is_grad_enabled() and
            isinstance(value, torch.Tensor) and value.is_cuda and
            torch.is_autocast_enabled())


def _float_tree(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_float_tree(v) for v in value)
    if isinstance(value, list):
        return [_float_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _float_tree(v) for k, v in value.items()}
    return value


def _fp32_call(module, original):
    def forward(*args, **kwargs):
        # These heads/embeddings always receive at least one positional tensor
        # or a tensor tree. The outer autocast wrapper is inference-only.
        if module.training or torch.is_grad_enabled() or not torch.is_autocast_enabled():
            return original(*args, **kwargs)
        with torch.autocast('cuda', enabled=False):
            return original(*_float_tree(args), **_float_tree(kwargs))
    return forward


def _fp32_entry(module, original):
    def forward(value, *args, **kwargs):
        if _active(module, value):
            value = value.float()
        return original(value, *args, **kwargs)
    return forward


def _fp32_exit(module, original):
    def forward(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        return result.float() if _active(module, value) else result
    return forward


def _autocast_call(module, original):
    def forward(value, *args, **kwargs):
        if (module.training or torch.is_grad_enabled() or
                not isinstance(value, torch.Tensor) or not value.is_cuda):
            return original(value, *args, **kwargs)
        # Casting each parameter within the graph avoids relying on the
        # lifetime of autocast's eager weight cache across graph captures.
        with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            return original(value, *args, **kwargs)
    return forward


def install_backbone_amp(diffusion):
    """Install after fusions; never convert model parameters to low precision."""
    from unifolm_wma.modules.attention import (
        BasicTransformerBlock, SpatialTransformer, TemporalTransformer,
        CrossAttention)
    from unifolm_wma.modules.networks.wma_model import ResBlock, TemporalConvBlock

    if getattr(diffusion, '_backbone_amp_installed', False):
        raise ValueError('Backbone AMP already installed')
    if any(p.is_floating_point() and p.dtype != torch.float32
           for p in diffusion.parameters()):
        raise ValueError('Backbone AMP requires original FP32 parameters')
    protected_names = ('action_unet', 'state_unet', 'out', 'time_embed')
    protected = [getattr(diffusion, name) for name in protected_names]
    if hasattr(diffusion, 'fps_embedding'):
        protected.append(diffusion.fps_embedding)
    excluded = {id(m) for root in protected for m in root.modules()}
    counts = {'residual_entries': 0, 'fp32_norms': 0,
              'fp32_heads_and_embeddings': len(protected), 'attention': 0}
    for module in diffusion.modules():
        if id(module) in excluded:
            continue
        if isinstance(module, (ResBlock, TemporalConvBlock, BasicTransformerBlock,
                               SpatialTransformer, TemporalTransformer)):
            module.forward = _fp32_entry(module, module.forward)
            counts['residual_entries'] += 1
        if isinstance(module, ResBlock):
            # Channel-changing skip projections autocast too. Materialize
            # their result in FP32 before adding the residual branch.
            skip = module.skip_connection
            skip.forward = _fp32_exit(skip, skip.forward)
        if isinstance(module, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            module.forward = _fp32_call(module, module.forward)
            counts['fp32_norms'] += 1
        if type(module) is CrossAttention:
            # Native Linear autocast, including condition-cache refresh, must
            # agree with the dtype expected by xFormers. Keep master weights.
            module._inference_attention_dtype = torch.float16
            module._inference_attention_self_only = False
            module._inference_projection_dtype = torch.float32
            counts['attention'] += 1
    for module in protected:
        module.forward = _fp32_call(module, module.forward)
        if hasattr(module, 'prepare_condition_cache'):
            module.prepare_condition_cache = _fp32_call(
                module, module.prepare_condition_cache)
    diffusion.dtype = torch.float32
    diffusion._fp32_feedback_heads = True
    diffusion._fp32_condition_embeddings = True
    diffusion.forward = _autocast_call(diffusion, diffusion.forward)
    diffusion.prepare_condition_cache = _autocast_call(
        diffusion, diffusion.prepare_condition_cache)
    diffusion._backbone_amp_installed = True
    return counts
