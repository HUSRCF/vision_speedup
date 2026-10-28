"""Experimental three-product TF32 approximation with FP32 outputs.

Each operand is split into a TF32-representable high part and an FP32
residual. Compute high*high + high*low + low*high; low*low is omitted.
This is not bitwise FP32 and requires end-to-end video quality validation.
"""
from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F


def split_tf32(value):
    bits = value.view(torch.int32)
    # Round to nearest, ties to even, retaining the FP32 exponent range.
    high = ((bits + 0xFFF + ((bits >> 13) & 1)) & -8192).view(torch.float32)
    return high, value - high


@contextmanager
def tensor_core_math():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def three_product(x, weight_parts, operation, bias=None):
    x_high, x_low = split_tf32(x)
    w_high, w_low = weight_parts
    with tensor_core_math():
        correction = operation(x_high, w_low)
        correction.add_(operation(x_low, w_high))
        result = operation(x_high, w_high)
        result.add_(correction)
    if bias is not None:
        result.add_(bias)
    return result


def make_forward(module, *, min_matmul_macs=500_000_000):
    """Build an inference-only forward; keep the original registered weights."""
    original = module.forward
    parts = None
    signature = None
    linear = isinstance(module, torch.nn.Linear)

    def forward(x):
        nonlocal parts, signature
        if (module.training or torch.is_grad_enabled() or x.dtype != torch.float32
                or module.weight.dtype != torch.float32 or not x.is_cuda):
            return original(x)
        if linear and x.numel() * module.out_features < min_matmul_macs:
            # Tiny conditioning projections cannot amortize three launches.
            return original(x)
        current = (module.weight.data_ptr(), module.weight._version,
                   module.weight.device, module.weight.dtype)
        if current != signature:
            parts = split_tf32(module.weight.detach())
            signature = current
        if linear:
            operation = lambda a, b: F.linear(a, b, None)
            bias = module.bias
        else:
            operation = lambda a, b: module._conv_forward(a, b, None)
            bias = None if module.bias is None else module.bias.reshape(
                1, -1, *((1,) * (x.ndim - 2)))
        return three_product(x, parts, operation, bias)

    return forward


def install_tf32x3(module, *, convolutions=True, linears=True):
    counts = {}
    for child in module.modules():
        # The measured 320 -> 2560 feed-forward projection was slower with
        # three products. Restrict the experiment to compute-heavy shapes.
        eligible = ((convolutions and isinstance(child, (torch.nn.Conv2d, torch.nn.Conv3d))
                     and min(child.in_channels, child.out_channels) >= 64
                     and math.prod(child.kernel_size) > 1)
                    or (linears and isinstance(child, torch.nn.Linear)
                        and min(child.in_features, child.out_features) >= 640))
        if eligible:
            if getattr(child, "_tf32x3_installed", False):
                raise RuntimeError("TF32x3 already installed")
            child.forward = make_forward(child)
            child._tf32x3_installed = True
            name = type(child).__name__
            counts[name] = counts.get(name, 0) + 1
    return counts
