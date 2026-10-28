"""World C640 concat F2 scopes, using the unchanged three-product kernels."""
import torch
from .triton_winograd_f2 import transform_filter_f2, winograd_f2_conv2d


def eligible(module, scope='all'):
    if scope not in {'all', '960', '1280', '1920'}:
        raise ValueError('Invalid concat F2 scope')
    return (isinstance(module, torch.nn.Conv2d) and module.kernel_size == (3, 3)
            and module.stride == (1, 1) and module.padding == (1, 1)
            and module.dilation == (1, 1) and module.groups == 1
            and module.padding_mode == 'zeros' and module.out_channels == 640
            and module.in_channels in {960, 1280, 1920}
            and (scope == 'all' or module.in_channels == int(scope)))


def make_forward(module, scope='all'):
    original = module.forward
    signature, parts = None, None

    def forward(value):
        nonlocal signature, parts
        weight = module.weight
        if (not eligible(module, scope) or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32 or not value.is_cuda
                or value.dtype != torch.float32 or value.ndim != 4
                or tuple(value.shape) != (16, module.in_channels, 20, 32)
                or not value.is_contiguous() or weight.dtype != torch.float32
                or weight.device != value.device or not weight.is_contiguous()
                or (module.bias is not None and (module.bias.dtype != torch.float32
                    or module.bias.device != value.device or not module.bias.is_contiguous()))):
            return original(value)
        try:
            version = weight._version
        except RuntimeError:
            return original(value)
        key = weight.data_ptr(), version, tuple(weight.shape), weight.dtype, weight.device
        if key != signature:
            if torch.cuda.is_current_stream_capturing():
                return original(value)
            reuse = (parts if parts is not None and parts[0].device == weight.device
                     and parts[0].shape == (16, module.out_channels, module.in_channels) else None)
            parts = transform_filter_f2(weight, out=reuse)
            signature = key
        return winograd_f2_conv2d(value, parts, module.bias, block_m=64)

    return forward


def install_world_concat_f2(diffusion, scope):
    selected = [module for module in diffusion.modules() if eligible(module, scope)]
    for module in selected:
        if any(getattr(module, marker, False) for marker in
               ('_world_concat_f2_installed', '_winograd_installed',
                '_winograd_f2_installed', '_packed_spatial_installed')):
            raise ValueError('Concat F2 requires a previously native C640 convolution')
    for module in selected:
        module.forward = make_forward(module, scope)
        module._world_concat_f2_installed = True
    return len(selected)
