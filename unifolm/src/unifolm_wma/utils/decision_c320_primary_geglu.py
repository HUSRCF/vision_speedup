"""Decision C320 GEGLU branch precision; preserve accepted world wrapper."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_primary_narrow_geglu import primary_narrow_geglu


def make_forward(module, mode, block_n):
    if mode not in {'both', 'value', 'gate'} or block_n not in {32, 64}:
        raise ValueError('Invalid decision C320 GEGLU experiment')
    original, projection = module.forward, module.proj
    signature, parts = None, None

    def forward(value):
        nonlocal signature, parts
        if (module.training or projection.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or not torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32 or not value.is_cuda
                or value.dtype != torch.float32 or value.ndim < 1 or not value.is_contiguous()
                or value.shape[-1] != 320 or value.numel() // 320 != 40960
                or tuple(projection.weight.shape) != (2560, 320)
                or projection.weight.dtype != torch.float32
                or projection.weight.device != value.device
                or not projection.weight.is_contiguous()
                or (projection.bias is not None and (projection.bias.shape != (2560,)
                    or projection.bias.dtype != torch.float32
                    or projection.bias.device != value.device
                    or not projection.bias.is_contiguous()))):
            return original(value)
        weight = projection.weight
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
        return primary_narrow_geglu(value, parts, projection.bias, mode=mode, block_n=block_n)

    return forward


def install_decision_c320_primary_geglus(diffusion, mode, block_n):
    from unifolm_wma.modules.attention import GEGLU
    count = 0
    for module in diffusion.modules():
        if (isinstance(module, GEGLU) and module.proj.in_features == 320
                and module.proj.out_features == 2560):
            if (not getattr(module, '_decision_half_geglu_installed', False)
                    or not getattr(module, '_world_prepacked_narrow_geglu_installed', False)
                    or getattr(module, '_decision_c320_primary_geglu_installed', False)):
                raise ValueError('C320 primary GEGLU requires accepted decision/world wrappers')
            module.forward = make_forward(module, mode, block_n)
            module._decision_c320_primary_geglu_installed = True
            count += 1
    return count
