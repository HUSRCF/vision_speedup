"""Protect world GEGLU while using scaled three-Half products in decision."""
import torch
from .triton_half_compensated_geglu import half_compensated_geglu


def make_forward(module):
    original = module.forward
    projection = module.proj

    def forward(value):
        if (module.training or projection.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or not torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype != torch.float32
                or not value.is_contiguous() or value.ndim < 1
                or value.shape[-1] != 320 or value.numel()//320 != 40960
                or projection.weight.dtype != torch.float32
                or projection.weight.device != value.device
                or not projection.weight.is_contiguous()
                or (projection.bias is not None and (projection.bias.dtype != torch.float32
                    or projection.bias.device != value.device or not projection.bias.is_contiguous()))):
            return original(value)
        return half_compensated_geglu(value, projection.weight, projection.bias,
            block_m=64, block_n=32, group_m=16)

    return forward


def install_decision_half_geglus(diffusion):
    from unifolm_wma.modules.attention import GEGLU
    count = 0
    for module in diffusion.modules():
        if (isinstance(module, GEGLU) and module.proj.in_features == 320
                and module.proj.out_features == 2560):
            if (not getattr(module, '_compensated_geglu_installed', False)
                    or getattr(module, '_decision_half_geglu_installed', False)):
                raise ValueError('Decision Half GEGLU requires accepted C320 fused TF32 route')
            module.forward = make_forward(module)
            module._decision_half_geglu_installed = True
            count += 1
    return count
