"""Use exact initial BK32 fusion only in the accepted world-stage FFN."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_initial_bk32_geglu import initial_bk32_geglu


def make_forward(module):
    original,projection=module.forward,module.proj
    signature,parts=None,None

    def forward(value):
        nonlocal signature,parts
        weight=projection.weight
        if (module.training or projection.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32 or not value.is_cuda
                or value.dtype!=torch.float32 or value.ndim<1 or not value.is_contiguous()
                or value.shape[-1]!=512 or value.numel()//512!=40960
                or tuple(weight.shape)!=(4096,512) or weight.dtype!=torch.float32
                or weight.device!=value.device or not weight.is_contiguous()
                or (projection.bias is not None and (projection.bias.shape!=(4096,)
                    or projection.bias.dtype!=torch.float32 or projection.bias.device!=value.device
                    or not projection.bias.is_contiguous()))):
            return original(value)
        try:version=weight._version
        except RuntimeError:return original(value)
        key=weight.data_ptr(),version,tuple(weight.shape),weight.dtype,weight.device
        if key!=signature:
            if torch.cuda.is_current_stream_capturing():return original(value)
            reuse=(parts if parts is not None and parts[0].shape==weight.shape
                   and parts[0].device==weight.device else None)
            parts=pack_half_weight(weight,out=reuse)
            signature=key
        return initial_bk32_geglu(value,parts,projection.bias,tile=(64,32,32,4,2))

    return forward


def install_initial_bk32_native_geglu(diffusion):
    from unifolm_wma.modules.attention import GEGLU
    selected=[]
    for name,module in diffusion.named_modules():
        if (not name.startswith('init_attn.') or not isinstance(module,GEGLU)
                or (module.proj.in_features,module.proj.out_features)!=(512,4096)):
            continue
        if (not getattr(module.proj,'_initial_world_linear_installed',False)
                or not getattr(module.proj,'_initial_decision_linear_installed',False)
                or getattr(module,'_initial_bk32_native_geglu_installed',False)
                or getattr(module,'_initial_prepacked_geglu_installed',False)
                or getattr(module,'_initial_primary_geglu_installed',False)):
            raise ValueError('Expected accepted initial512 stage wrappers exactly once')
        selected.append(module)
    for module in selected:
        module.forward=make_forward(module)
        module._initial_bk32_native_geglu_installed=True
    return len(selected)
