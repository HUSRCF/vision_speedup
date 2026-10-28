"""Opt-in short-block compensated GEMM for the two small spatial convolutions.

Weights and outputs stay FP32; three TF32 products approximate each FP32
product. Full trajectory PSNR validation is required before adopting a run.
"""
import torch
import torch.nn.functional as F
from .triton_compensated_linear import compensated_linear


def eligible(module):
    return (isinstance(module, torch.nn.Conv2d) and
            module.in_channels in (1280, 2560) and module.out_channels == 1280 and
            module.kernel_size == (3, 3) and module.stride == (1, 1) and
            module.padding == (1, 1) and module.dilation == (1, 1) and
            module.groups == 1 and module.padding_mode == 'zeros')


def make_forward(module):
    original = module.forward

    def forward(value):
        if (not eligible(module) or module.training or torch.is_grad_enabled() or
                not value.is_cuda or value.dtype != torch.float32 or
                value.ndim != 4 or value.shape[0] == 0 or
                value.shape[1] != module.in_channels or value.shape[2:] != (5, 8) or
                not value.is_contiguous() or torch.backends.cudnn.allow_tf32 or
                module.weight.dtype != torch.float32 or
                module.weight.device != value.device or not module.weight.is_contiguous() or
                (module.bias is not None and (module.bias.dtype != torch.float32 or
                                             module.bias.device != value.device))):
            return original(value)
        batch, channels, height, width = value.shape
        patches = F.unfold(value, (3, 3), padding=1)
        matrix = patches.transpose(1, 2).contiguous().view(
            batch * height * width, channels * 9)
        result = compensated_linear(matrix, module.weight.flatten(1), module.bias)
        return result.view(batch, height * width, module.out_channels).transpose(
            1, 2).reshape(batch, module.out_channels, height, width).contiguous()

    return forward


def install_compensated_spatial_convs(model):
    count = 0
    for module in model.modules():
        if eligible(module):
            if getattr(module, '_compensated_spatial_installed', False):
                raise RuntimeError('Compensated spatial convolution already installed')
            module.forward = make_forward(module)
            module._compensated_spatial_installed = True
            count += 1
    return count
