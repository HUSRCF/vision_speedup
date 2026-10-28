"""Decision-only FFN using its already-required Half intermediate directly."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_half_hidden_initial_ffn import half_hidden_ffn


def make_forward(module):
    original = module.forward
    layers = (module.net[0].proj,module.net[2])
    signatures,parts = [None,None],[None,None]

    def forward(value):
        if (module.training or any(layer.training for layer in module.net) or layers[0].training
                or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or not torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or module._inference_dtype!=torch.float32 or not value.is_cuda
                or value.dtype!=torch.float32 or value.ndim<1 or not value.is_contiguous()
                or value.shape[-1]!=512 or value.numel()//512!=40960):
            return original(value)
        for index,(layer,shape) in enumerate(zip(layers,[(4096,512),(512,2048)])):
            weight = layer.weight
            if (tuple(weight.shape)!=shape or weight.dtype!=torch.float32
                    or weight.device!=value.device or not weight.is_contiguous()
                    or (layer.bias is not None and (layer.bias.shape!=(shape[0],)
                        or layer.bias.dtype!=torch.float32 or layer.bias.device!=value.device
                        or not layer.bias.is_contiguous()))):
                return original(value)
            try:version=weight._version
            except RuntimeError:return original(value)
            key=weight.data_ptr(),version,tuple(weight.shape),weight.dtype,weight.device
            if signatures[index]!=key:
                if torch.cuda.is_current_stream_capturing():return original(value)
                reuse=(parts[index] if parts[index] is not None and
                    parts[index][0].shape==weight.shape and parts[index][0].device==weight.device else None)
                parts[index]=pack_half_weight(weight,out=reuse)
                signatures[index]=key
        return half_hidden_ffn(value,parts[0][0],parts[1][0],layers[0].bias,layers[1].bias)

    return forward


def install_half_hidden_initial_ffn(diffusion):
    from unifolm_wma.modules.attention import FeedForward,GEGLU
    count=0
    for name,module in diffusion.named_modules():
        if (not name.startswith('init_attn.') or not isinstance(module,FeedForward)
                or len(module.net)!=3 or not isinstance(module.net[0],GEGLU)
                or type(module.net[1]) is not torch.nn.Dropout
                or not isinstance(module.net[2],torch.nn.Linear)
                or (module.net[0].proj.in_features,module.net[0].proj.out_features)!=(512,4096)
                or (module.net[2].in_features,module.net[2].out_features)!=(2048,512)):
            continue
        if (getattr(module,'_half_hidden_initial_ffn_installed',False)
                or getattr(module.net[0],'_initial_primary_geglu_installed',False)
                or getattr(module.net[0],'_initial_prepacked_geglu_installed',False)
                or any(not getattr(layer,'_decision_initial_primary_installed',False)
                       or not getattr(layer,'_initial_world_linear_installed',False)
                       for layer in (module.net[0].proj,module.net[2]))):
            raise ValueError('Half hidden FFN requires accepted initial primary pair')
        module.forward=make_forward(module)
        module._half_hidden_initial_ffn_installed=True
        count+=1
    return count
