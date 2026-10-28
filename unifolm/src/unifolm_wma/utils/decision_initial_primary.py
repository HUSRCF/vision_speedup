"""Opt-in one-product initial512 projections, confined to the decision stage."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_primary_narrow_linear import primary_narrow_linear


PAIRS = {(512, 512), (512, 4096), (2048, 512)}


def make_forward(module):
    original = module.forward
    signature, parts = None, None
    inputs, outputs = module.in_features, module.out_features

    def forward(value):
        nonlocal signature, parts
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or not torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype != torch.float32 or value.ndim < 1
                or not value.is_contiguous() or value.shape[-1] != inputs
                or value.numel() // inputs != 40960
                or tuple(module.weight.shape) != (outputs, inputs)
                or module.weight.dtype != torch.float32 or module.weight.device != value.device
                or not module.weight.is_contiguous()
                or (module.bias is not None and (module.bias.shape != (outputs,)
                    or module.bias.dtype != torch.float32 or module.bias.device != value.device
                    or not module.bias.is_contiguous()))):
            return original(value)
        weight = module.weight
        try:
            version = weight._version
        except RuntimeError:
            return original(value)
        key = weight.data_ptr(), version, tuple(weight.shape), weight.dtype, weight.device
        if key != signature:
            if torch.cuda.is_current_stream_capturing():
                return original(value)
            reuse = (parts if parts is not None and parts[0].shape == weight.shape
                     and parts[0].device == weight.device else None)
            parts = pack_half_weight(weight, out=reuse)
            signature = key
        return primary_narrow_linear(value, parts[0], module.bias, tile=(64, 128, 32, 4))

    return forward


def install_decision_initial_primary(diffusion, scope):
    if scope not in {'projections', 'ffn', 'all'}:
        raise ValueError('Invalid initial512 primary scope')
    count = 0
    for name, module in diffusion.named_modules():
        if not name.startswith('init_attn.') or not isinstance(module, torch.nn.Linear):
            continue
        pair = module.in_features, module.out_features
        if pair not in PAIRS or (scope == 'projections' and pair != (512, 512)):
            continue
        if scope == 'ffn' and pair == (512, 512):
            continue
        if (not getattr(module, '_initial_decision_linear_installed', False)
                or not getattr(module, '_initial_world_linear_installed', False)
                or getattr(module, '_decision_initial_primary_installed', False)):
            raise ValueError('Initial primary requires accepted decision and world wrappers')
        module.forward = make_forward(module)
        module._decision_initial_primary_installed = True
        count += 1
    return count
