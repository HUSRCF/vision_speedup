"""Inference candidate: collapse independent spatial axes of temporal Conv3d.

The operation and FP32 dtype are preserved. cuDNN can select a different
algorithm for the equivalent 2-D descriptor, so video validation is required.
"""
import torch
import torch.nn.functional as F


def make_temporal_conv2d_forward(module, *, preserve_tf32=False):
    if not isinstance(module, torch.nn.Conv3d):
        raise TypeError("Expected Conv3d")
    if (module.kernel_size[1:] != (1, 1)
            or module.stride[1:] != (1, 1)
            or module.padding[1:] != (0, 0)
            or module.dilation[1:] != (1, 1)
            or module.padding_mode != "zeros"):
        raise ValueError("Only spatially independent temporal Conv3d is supported")
    original = module.forward

    def forward(value):
        if module.training or torch.is_grad_enabled():
            return original(value)
        if preserve_tf32 and torch.backends.cudnn.allow_tf32:
            return original(value)
        height, width = value.shape[-2:]
        output = F.conv2d(
            value.flatten(3), module.weight.squeeze(-1), module.bias,
            (module.stride[0], 1), (module.padding[0], 0),
            (module.dilation[0], 1), module.groups)
        return output.unflatten(3, (height, width))

    return forward
