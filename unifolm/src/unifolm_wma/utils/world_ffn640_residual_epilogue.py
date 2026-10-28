"""World norm3->GEGLU packing fusion with explicit per-call Half inputs."""
import torch
from .triton_residual_layer_norm import fused_residual_layer_norm
from .triton_half_compensated_linear import pack_half_weight
from .triton_zero_weight_low_geglu import residual_count
from .residual_norm_half_pack import residual_norm_half_pack
from .geglu_from_input_parts import geglu_from_input_parts
from .ffn640_residual_epilogue import ffn640_residual_epilogue


ROWS={640:{10240}}


def compatible(block):
    from .world_norm3_packed_geglu import compatible as norm_compatible
    return (norm_compatible(block) and block.norm3.normalized_shape==(640,)
        and getattr(block,'_world_norm3_packed_geglu_installed',False)
        and getattr(block.ff.net[2],'_tf32x3_installed',False)
        and getattr(block.ff.net[2],'_world_two_product_roles_installed',False))


def make_forward(block):
    original=block._forward
    def forward(value,context=None,mask=None):
        if not compatible(block):return original(value,context=context,mask=mask)
        ge=block.ff.net[0];p=ge.proj;norm=block.norm3;c=p.in_features
        final=block.ff.net[2]
        parameters=[final.weight,*([final.bias] if final.bias is not None else []),p.weight,*([p.bias] if p.bias is not None else []),norm.weight,norm.bias]
        if (any(m.training for m in block.modules())
                or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype!=torch.float32 or value.ndim!=3
                or not value.is_contiguous() or value.shape[-1]!=c
                or value.numel()//c not in ROWS[c]
                or any(t is None or t.dtype!=torch.float32 or t.device!=value.device
                       or not t.is_contiguous() for t in parameters)):
            return original(value,context=context,mask=mask)
        w=p.weight
        try:key=(w.data_ptr(),w._version,tuple(w.shape),w.dtype,w.device)
        except RuntimeError:return original(value,context=context,mask=mask)
        fw=final.weight
        try:final_key=(fw.data_ptr(),fw._version,tuple(fw.shape),fw.dtype,fw.device)
        except RuntimeError:return original(value,context=context,mask=mask)
        final_cache=getattr(final,'_ffn640_residual_weight_cache',None)
        if (final_cache is None or final_cache[0]!=final_key) and torch.cuda.is_current_stream_capturing():
            return original(value,context=context,mask=mask)
        cache=getattr(ge,'_norm3_geglu_weight_cache',None)
        if cache is None or cache[0]!=key:
            if torch.cuda.is_current_stream_capturing():
                return original(value,context=context,mask=mask)
            reuse=(cache[1] if cache is not None and cache[1][0].shape==w.shape
                   and cache[1][0].device==w.device else None)
            parts=pack_half_weight(w,out=reuse)
            count=residual_count(parts,refresh=True)
            if count is None:return original(value,context=context,mask=mask)
            cache=key,parts,count
            ge._norm3_geglu_weight_cache=cache
        if final_cache is None or final_cache[0]!=final_key:
            reuse=(final_cache[1] if final_cache is not None and final_cache[1][0].shape==fw.shape
                   and final_cache[1][0].device==fw.device else None)
            final_cache=final_key,pack_half_weight(fw,out=reuse)
            final._ffn640_residual_weight_cache=final_cache
        attended=block.attn1(block.norm1(value),
            context=context if block.disable_self_attn else None,mask=mask)
        residual,normalized=fused_residual_layer_norm(attended,value,block.norm2)
        attended=block.attn2(normalized,context=context,mask=mask)
        if (attended.shape!=residual.shape or attended.device!=residual.device
                or attended.dtype!=torch.float32 or not attended.is_contiguous()
                or not residual.is_contiguous()):
            residual,normalized=fused_residual_layer_norm(attended,residual,norm)
            return block.ff(normalized)+residual
        block._world_norm3_packed_geglu_calls=getattr(block,'_world_norm3_packed_geglu_calls',0)+1
        residual,high,low=residual_norm_half_pack(attended,residual,norm)
        hidden=geglu_from_input_parts((high,low),cache[1],cache[2],p.bias,
                                     shape=(*value.shape[:-1],c*4))
        block._world_ffn640_residual_epilogue_calls=getattr(block,'_world_ffn640_residual_epilogue_calls',0)+1
        return ffn640_residual_epilogue(block.ff.net[1](hidden),final_cache[1],residual,final.bias)
    return forward


def install_world_ffn640_residual_epilogue(diffusion,scope='all'):
    from unifolm_wma.modules.attention import BasicTransformerBlock
    if scope not in {'all','c320','c640','c1280'}:raise ValueError('Invalid norm3 GEGLU packing scope')
    selected=[]
    for block in diffusion.modules():
        if not isinstance(block,BasicTransformerBlock):continue
        c=block.norm3.normalized_shape[0]
        if c not in ROWS or scope not in {'all','c'+str(c)}:continue
        if not compatible(block) or getattr(block,'_world_ffn640_residual_epilogue_installed',False):
            raise ValueError('Expected accepted world residual norm and GEGLU configuration')
        selected.append(block)
    for block in selected:
        block._forward=make_forward(block)
        block._world_ffn640_residual_epilogue_installed=True
    return len(selected)
