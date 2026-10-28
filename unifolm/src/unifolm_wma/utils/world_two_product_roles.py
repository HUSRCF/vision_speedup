"""Separate world projection/FFN precision experiments with guarded stage scope."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_two_product_linear import two_product_linear


def make_forward(module, *, protect, rows):
    original = module.forward
    inputs, outputs = module.in_features, module.out_features
    signature, parts = None, None

    def forward(value):
        nonlocal signature, parts
        weight = module.weight
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype != torch.float32 or value.ndim < 1
                or not value.is_contiguous() or value.shape[-1] != inputs
                or value.numel() // inputs != rows or tuple(weight.shape) != (outputs, inputs)
                or weight.dtype != torch.float32 or weight.device != value.device
                or not weight.is_contiguous()
                or (module.bias is not None and (module.bias.shape != (outputs,)
                    or module.bias.dtype != torch.float32 or module.bias.device != value.device
                    or not module.bias.is_contiguous()))):
            return original(value)
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
        return two_product_linear(value, parts, module.bias, protect=protect)

    return forward


def install_world_two_product_roles(diffusion, scope, protect):
    if scope not in {'qk320', 'output640', 'ffn640'} or protect not in {'input', 'weight'}:
        raise ValueError('Invalid world two-product role/protection')
    selected = []
    for name, module in diffusion.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if scope == 'qk320':
            match = name.endswith(('.to_q', '.to_k'))
            shape, rows = (320, 320), 40960
        elif scope == 'output640':
            match = name.endswith(('.to_out.0', '.proj_in', '.proj_out'))
            shape, rows = (640, 640), 10240
        else:
            match = name.endswith('.ff.net.2')
            shape, rows = (640, 2560), 10240
        if not match or (module.out_features, module.in_features) != shape:
            continue
        if (not getattr(module, '_tf32x3_installed', False)
                or (scope == 'qk320' and
                    not getattr(module, '_decision_primary_narrow_installed', False))
                or getattr(module, '_world_primary_narrow_installed', False)
                or getattr(module, '_world_two_product_output_installed', False)
                or getattr(module, '_world_two_product_roles_installed', False)):
            raise ValueError('Expected protected accepted world role: ' + name)
        selected.append((module, rows))
    for module, rows in selected:
        module.forward = make_forward(module, protect=protect, rows=rows)
        module._world_two_product_roles_installed = True
    return len(selected)
