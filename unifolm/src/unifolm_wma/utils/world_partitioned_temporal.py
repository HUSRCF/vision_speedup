"""Stage-guarded temporal residual dispatch over the accepted conv wrapper."""
import torch
from .triton_compensated_temporal_conv import eligible_temporal, temporal_weight_parts
from .triton_zero_weight_low_linear import weight_low_count
from .partitioned_temporal import partitioned_temporal_conv


def make_forward(module, scope='implicit'):
    original = module.forward
    signature, parts, count = None, None, None
    selected = {(320, 40, 64), (640, 20, 32), (1280, 10, 16), (1280, 5, 8)}

    def forward(value):
        nonlocal signature, parts, count
        weight = module.weight
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype != torch.float32 or value.ndim != 5
                or value.shape[0] != 1 or value.shape[2] != 16
                or (value.shape[1], *value.shape[-2:]) not in selected
                or (scope == 'implicit' and value.shape[1] == 1280)
                or (scope == 'packed' and value.shape[1] != 1280)
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
            reuse = parts if parts is not None and parts[0].device == weight.device else None
            parts = temporal_weight_parts(weight, out=reuse)
            count = weight_low_count(parts, out=count if count is not None and count.device == weight.device else None)
            signature = key
        return partitioned_temporal_conv(value, parts, count, module.bias,
            group_m=8 if value.shape[1] == 320 else 16)

    return forward


def install_world_partitioned_temporal(diffusion, scope='implicit'):
    if scope != 'implicit':
        raise ValueError('Invalid temporal zero-weight-residual scope')
    selected = []
    for module in diffusion.modules():
        if not eligible_temporal(module):
            continue
        if getattr(module, '_world_partitioned_temporal_installed', False):
            raise ValueError('Duplicate temporal zero-weight-residual installation')
        if getattr(module.forward, '__qualname__', '') != 'make_compensated_temporal_forward.<locals>.forward':
            continue
        if ((scope == 'implicit' and module.in_channels == 1280)
                or (scope == 'packed' and module.in_channels != 1280)):
            continue
        selected.append(module)
    for module in selected:
        module.forward = make_forward(module, scope)
        module._world_partitioned_temporal_installed = True
    return len(selected)
