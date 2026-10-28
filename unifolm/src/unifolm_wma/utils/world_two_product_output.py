"""World-only C320 output projections; all other installed paths are preserved."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_two_product_linear import two_product_linear


def make_forward(module, protect):
    original = module.forward
    signature, parts = None, None

    def forward(value):
        nonlocal signature, parts
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype != torch.float32 or value.ndim < 1
                or not value.is_contiguous() or value.shape[-1] != 320
                or value.numel() // 320 != 40960 or tuple(module.weight.shape) != (320, 320)
                or module.weight.dtype != torch.float32 or module.weight.device != value.device
                or not module.weight.is_contiguous()
                or (module.bias is not None and (module.bias.shape != (320,)
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
        return two_product_linear(value, parts, module.bias, protect=protect)

    return forward


def install_world_two_product_output(diffusion, protect):
    if protect not in {'input', 'weight'}:
        raise ValueError('Invalid world two-product protection')
    selected = [(name, module) for name, module in diffusion.named_modules()
                if name.endswith(('.to_out.0', '.proj_in', '.proj_out'))
                and isinstance(module, torch.nn.Linear)
                and module.in_features == 320 and module.out_features == 320]
    for name, module in selected:
        if (not getattr(module, '_decision_primary_narrow_installed', False)
                or not getattr(module, '_tf32x3_installed', False)
                or getattr(module, '_world_primary_narrow_installed', False)
                or getattr(module, '_world_two_product_output_installed', False)):
            raise ValueError('Expected accepted decision and protected world output: ' + name)
    for _, module in selected:
        module.forward = make_forward(module, protect)
        module._world_two_product_output_installed = True
    return len(selected)
