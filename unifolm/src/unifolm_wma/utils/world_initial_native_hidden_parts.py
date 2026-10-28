"""World initial FFN with fresh packed intermediates and the original BK32 math."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .initial_native_hidden_parts_pipeline import initial_native_hidden_parts,initial_packed_final


def compatible(module):
    from unifolm_wma.modules.attention import GEGLU
    return (len(module.net)==3 and isinstance(module.net[0],GEGLU)
        and type(module.net[1]) is torch.nn.Dropout
        and isinstance(module.net[2],torch.nn.Linear)
        and (module.net[0].proj.in_features,module.net[0].proj.out_features)==(512,4096)
        and (module.net[2].in_features,module.net[2].out_features)==(2048,512)
        and getattr(module,'_half_hidden_initial_ffn_installed',False)
        and getattr(module.net[0],'_initial_bk32_native_geglu_installed',False)
        and all(getattr(p,'_initial_world_linear_installed',False)
                and getattr(p,'_decision_initial_primary_installed',False)
                for p in (module.net[0].proj,module.net[2])))


def make_forward(module):
    original=module.forward
    def forward(value):
        if not compatible(module):return original(value)
        if (any(m.training for m in module.modules()) or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32
                or module._inference_dtype!=torch.float32 or not value.is_cuda
                or value.dtype!=torch.float32 or value.ndim<1 or not value.is_contiguous()
                or value.shape[-1]!=512 or value.numel()//512!=40960):
            return original(value)
        layers=(module.net[0].proj,module.net[2]);keys=[]
        for layer,shape in zip(layers,[(4096,512),(512,2048)]):
            w=layer.weight;b=layer.bias
            if (tuple(w.shape)!=shape or w.dtype!=torch.float32 or w.device!=value.device
                    or not w.is_contiguous() or (b is not None and (b.shape!=(shape[0],)
                        or b.dtype!=torch.float32 or b.device!=value.device or not b.is_contiguous()))):
                return original(value)
            try:key=(w.data_ptr(),w._version,tuple(w.shape),w.dtype,w.device)
            except RuntimeError:return original(value)
            keys.append(key)
        cache=getattr(module,'_world_initial_native_hidden_parts_weight_cache',None)
        if (cache is None or cache[0]!=tuple(keys)) and torch.cuda.is_current_stream_capturing():
            return original(value)
        parts=[]
        for index,layer in enumerate(layers):
            if cache is not None and cache[0][index]==keys[index]:
                part=cache[1][index]
            else:
                reuse=cache[1][index] if cache is not None else None
                if reuse is not None and (reuse[0].shape!=layer.weight.shape or reuse[0].device!=layer.weight.device):reuse=None
                part=pack_half_weight(layer.weight,out=reuse)
            parts.append(part)
        module._world_initial_native_hidden_parts_weight_cache=tuple(keys),tuple(parts)
        hidden=initial_native_hidden_parts(value.view(-1,512),parts[0],layers[0].bias)
        output=initial_packed_final(hidden,parts[1],layers[1].bias)
        module._world_initial_native_hidden_parts_calls=getattr(module,'_world_initial_native_hidden_parts_calls',0)+1
        return output.view_as(value)
    return forward


def install_world_initial_native_hidden_parts(diffusion):
    from unifolm_wma.modules.attention import FeedForward
    selected=[]
    for name,module in diffusion.named_modules():
        if (not name.startswith('init_attn.') or not isinstance(module,FeedForward)
                or module.net[0].proj.in_features!=512):continue
        if not compatible(module) or getattr(module,'_world_initial_native_hidden_parts_installed',False):
            raise ValueError('Expected accepted initial world/decision FFN wrappers exactly once')
        selected.append(module)
    for module in selected:
        module.forward=make_forward(module)
        module._world_initial_native_hidden_parts_installed=True
    return len(selected)
