"""World deep5x8 compensated convolution; all other paths keep prior wrapper."""
import torch
from .triton_packed_spatial_conv import packed_spatial_conv, spatial_weight_parts
from .triton_winograd_f2 import eligible_f2


def make_forward(module, scope):
    if scope not in {'all', '1280'}:
        raise ValueError('Invalid packed spatial scope')
    original = module.forward
    signature, parts = None, None

    def forward(value):
        nonlocal signature, parts
        if (not eligible_f2(module) or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype != torch.float32 or value.ndim != 4
                or tuple(value.shape) != (16, module.in_channels, 5, 8)
                or not value.is_contiguous() or module.in_channels not in {1280, 2560}
                or (scope == '1280' and module.in_channels != 1280)
                or tuple(module.weight.shape) != (1280, module.in_channels, 3, 3)
                or module.weight.dtype != torch.float32
                or module.weight.device != value.device or not module.weight.is_contiguous()
                or (module.bias is not None and (module.bias.dtype != torch.float32
                    or module.bias.device != value.device or not module.bias.is_contiguous()))):
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
            reuse = (parts if parts is not None and parts[0].device == weight.device
                     and parts[0].shape == (1280, module.in_channels * 9) else None)
            parts = spatial_weight_parts(weight, out=reuse)
            signature = key
        return packed_spatial_conv(value, parts, module.bias, block_m=64,
            block_n=64, splits=1 if module.in_channels == 1280 else 4)

    return forward


def install_packed_spatial_convs(diffusion, scope):
    count = 0
    for module in diffusion.modules():
        if eligible_f2(module):
            if getattr(module, '_packed_spatial_installed', False):
                raise ValueError('Packed spatial convolution already installed')
            module.forward = make_forward(module, scope)
            module._packed_spatial_installed = True
            count += 1
    return count
