"""Fuse only the accepted decision initial512 primary projection and activation."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_inline_primary_geglu import inline_primary_geglu


TILE = (64,128,32,4,2)


def make_forward(module):
    original, projection = module.forward, module.proj
    signature, parts = None, None

    def forward(value):
        nonlocal signature, parts
        if (module.training or projection.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or not torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32 or not value.is_cuda
                or value.dtype!=torch.float32 or value.ndim<1 or not value.is_contiguous()
                or value.shape[-1]!=512 or value.numel()//512!=40960
                or tuple(projection.weight.shape)!=(4096,512)
                or projection.weight.dtype!=torch.float32
                or projection.weight.device!=value.device or not projection.weight.is_contiguous()
                or (projection.bias is not None and (projection.bias.shape!=(4096,)
                    or projection.bias.dtype!=torch.float32 or projection.bias.device!=value.device
                    or not projection.bias.is_contiguous()))):
            return original(value)
        weight = projection.weight
        try:
            version = weight._version
        except RuntimeError:
            return original(value)
        key = weight.data_ptr(),version,tuple(weight.shape),weight.dtype,weight.device
        if signature!=key:
            if torch.cuda.is_current_stream_capturing():
                return original(value)
            reuse = (parts if parts is not None and parts[0].shape==weight.shape
                     and parts[0].device==weight.device else None)
            parts = pack_half_weight(weight,out=reuse)
            signature = key
        return inline_primary_geglu(value,parts[0],projection.bias,tile=TILE,activation='native')

    return forward


def install_initial_primary_geglu(diffusion):
    from unifolm_wma.modules.attention import GEGLU
    count = 0
    for name,module in diffusion.named_modules():
        if (not name.startswith('init_attn.') or not isinstance(module,GEGLU)
                or (module.proj.in_features,module.proj.out_features)!=(512,4096)):
            continue
        if (not getattr(module.proj,'_initial_decision_linear_installed',False)
                or not getattr(module.proj,'_initial_world_linear_installed',False)
                or not getattr(module.proj,'_decision_initial_primary_installed',False)
                or getattr(module,'_initial_prepacked_geglu_installed',False)
                or getattr(module,'_initial_primary_geglu_installed',False)):
            raise ValueError('Initial GEGLU fusion requires the accepted primary FFN projection')
        module.forward = make_forward(module)
        module._initial_primary_geglu_installed = True
        count += 1
    return count
