"""Decision-only C320 projections with FP32 master, accumulation and output."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_primary_narrow_linear import primary_narrow_linear


def make_forward(module):
    original = module.forward
    signature, parts = None, None
    tile = (128, 64, 32, 4) if module.bias is None else (64, 128, 32, 4)

    def forward(value):
        nonlocal signature, parts
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or not torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
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
        return primary_narrow_linear(value, parts[0], module.bias, tile=tile)

    return forward


def install_decision_primary_narrow(diffusion, scope):
    if scope not in {'qkv', 'output', 'all'}:
        raise ValueError('Invalid decision C320 projection scope')
    count = 0
    for name, module in diffusion.named_modules():
        qkv = name.endswith(('.to_q', '.to_k', '.to_v'))
        output = name.endswith(('.to_out.0', '.proj_in', '.proj_out'))
        selected = (qkv or output) if scope == 'all' else qkv if scope == 'qkv' else output
        if (not selected or not isinstance(module, torch.nn.Linear)
                or module.in_features != 320 or module.out_features != 320):
            continue
        if (not getattr(module, '_tf32x3_installed', False)
                or getattr(module, '_decision_primary_narrow_installed', False)):
            raise ValueError('Decision C320 primary must wrap accepted compensated projection')
        module.forward = make_forward(module)
        module._decision_primary_narrow_installed = True
        count += 1
    return count
