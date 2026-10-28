"""World spatial attention and output projection with fresh packed intermediates."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_zero_weight_low_linear import weight_low_count
from .qkv_attention_hidden_projection import qkv_attention_hidden_projection

SHAPES={(16,2560,320),(16,640,640)}


def compatible(module):
    from .world_qkv_packed_attention import compatible as previous_compatible
    if (not previous_compatible(module) or not getattr(module,'_qkv_packed_attention_installed',False)
            or not isinstance(module.to_out,torch.nn.Sequential) or len(module.to_out)!=2
            or not isinstance(module.to_out[0],torch.nn.Linear)
            or type(module.to_out[1]) is not torch.nn.Dropout):return False
    channels=module.to_q.in_features;final=module.to_out[0]
    return ((final.in_features,final.out_features)==(channels,channels)
        and getattr(final,'_world_two_product_output_installed' if channels==320
                    else '_world_two_product_roles_installed',False))


def make_forward(module):
    original=module.forward
    def forward(value,context=None,mask=None):
        if not compatible(module):return original(value,context=context,mask=mask)
        projections=module.to_q,module.to_k,module.to_v;final=module.to_out[0]
        if (any(m.training for m in module.modules()) or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or context is not None or mask is not None
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype!=torch.float32 or not value.is_contiguous()
                or tuple(value.shape) not in SHAPES
                or tuple(final.weight.shape)!=(final.out_features,final.in_features)
                or any(p.weight.dtype!=torch.float32 or p.weight.device!=value.device
                       or not p.weight.is_contiguous() for p in (*projections,final))
                or (final.bias is not None and (final.bias.shape!=(final.out_features,)
                       or final.bias.dtype!=torch.float32 or final.bias.device!=value.device
                       or not final.bias.is_contiguous()))):
            return original(value,context=context,mask=mask)
        weights=tuple(p.weight for p in projections);fw=final.weight
        try:
            key=tuple((w.data_ptr(),w._version,tuple(w.shape),w.dtype,w.device) for w in weights)
            final_key=(fw.data_ptr(),fw._version,tuple(fw.shape),fw.dtype,fw.device)
        except RuntimeError:return original(value,context=context,mask=mask)
        channels=value.shape[-1];cache=getattr(module,'_world_zero_weight_low_qkv_cache',None)
        final_cache=getattr(final,'_attention_hidden_projection_weight_cache',None)
        if torch.cuda.is_current_stream_capturing() and (
                cache is None or cache[0]!=key or final_cache is None or final_cache[0]!=final_key):
            return original(value,context=context,mask=mask)
        if cache is None or cache[0]!=key:
            reuse=(cache is not None and cache[1].shape==(3*channels,channels) and cache[1].device==value.device)
            if reuse:
                combined=cache[1];torch.cat(weights,dim=0,out=combined)
                parts=pack_half_weight(combined,out=cache[2])
            else:
                combined=torch.cat(weights,dim=0);parts=pack_half_weight(combined)
            count=weight_low_count(parts,out=cache[3] if reuse else None)
            cache=key,combined,parts,count;module._world_zero_weight_low_qkv_cache=cache
        if final_cache is None or final_cache[0]!=final_key:
            reuse=(final_cache[1] if final_cache is not None and final_cache[1][0].shape==fw.shape
                   and final_cache[1][0].device==fw.device else None)
            final_cache=final_key,pack_half_weight(fw,out=reuse)
            final._attention_hidden_projection_weight_cache=final_cache
        result=qkv_attention_hidden_projection(value,cache[2],cache[3],final_cache[1],final.bias,scale=module.scale)
        module._world_attention_hidden_projection_calls=getattr(module,'_world_attention_hidden_projection_calls',0)+1
        return result
    return forward


def install_world_attention_hidden_projection(diffusion,scope='all'):
    from unifolm_wma.modules.attention import SpatialTransformer
    if scope not in {'all','c320','c640'}:raise ValueError('Invalid attention hidden projection scope')
    selected=[];seen=set()
    for parent in diffusion.modules():
        if not isinstance(parent,SpatialTransformer):continue
        for block in parent.transformer_blocks:
            module=block.attn1;channels=module.to_q.in_features
            if channels not in {320,640} or scope not in {'all','c'+str(channels)}:continue
            if id(module) in seen or not compatible(module) or getattr(module,'_world_attention_hidden_projection_installed',False):
                raise ValueError('Expected accepted spatial attention and protected output exactly once')
            seen.add(id(module));selected.append(module)
    for module in selected:
        module.forward=make_forward(module);module._world_attention_hidden_projection_installed=True
    return len(selected)
