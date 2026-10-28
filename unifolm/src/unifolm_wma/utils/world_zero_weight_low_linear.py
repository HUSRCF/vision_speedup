"""World C1280 linears with short-BK64 arithmetic and device residual dispatch."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_zero_weight_low_linear import zero_weight_low_linear, weight_low_count


def make_forward(module):
    original = module.forward
    inputs, outputs = module.in_features, module.out_features
    signature, parts, count = None, None, None

    def forward(value):
        nonlocal signature, parts, count
        weight = module.weight
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype != torch.float32 or value.ndim < 1
                or not value.is_contiguous() or value.shape[-1] != inputs
                or value.numel() // inputs not in {2560, 640}
                or tuple(weight.shape) != (outputs, inputs)
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
            reuse_count = count if count is not None and count.device == weight.device else None
            count = weight_low_count(parts, out=reuse_count)
            signature = key
        return zero_weight_low_linear(value, parts, count, module.bias)

    return forward


def install_world_zero_weight_low_linears(diffusion):
    selected = []
    for module in diffusion.modules():
        if (isinstance(module, torch.nn.Linear) and module.out_features == 1280
                and module.in_features in {1280, 5120}
                and getattr(module, '_tf32x3_installed', False)):
            if getattr(module, '_world_zero_weight_low_installed', False):
                raise ValueError('Duplicate world zero-weight-residual installation')
            selected.append(module)
    for module in selected:
        module.forward = make_forward(module)
        module._world_zero_weight_low_installed = True
    return len(selected)
