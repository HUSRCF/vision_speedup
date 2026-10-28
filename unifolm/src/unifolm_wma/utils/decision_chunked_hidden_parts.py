"""Decision C640 exact chunk512 FFN with shared weights and fresh Half hidden."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .decision_shared_ge_count import _refresh_shared_ge_count
from .triton_residual_layer_norm import fused_residual_layer_norm
from .residual_norm_half_pack import residual_norm_half_pack
from .cublas_compensated_linear import _READY,_stream_key
from .decision_chunked_hidden_pipeline import primary_geglu_hidden_parts,chunked_hidden_residual


def compatible(block):
    from .world_geglu_hidden_parts import compatible as world_compatible
    return (world_compatible(block)
        and getattr(block,'_world_geglu_hidden_parts_installed',False)
        and getattr(block.ff.net[0],'_decision_primary_geglu_installed',False)
        and getattr(block.ff.net[2],'_decision_vendor_ffn_installed',False)
        and not getattr(block.ff.net[2],'_decision_primary_ffn_installed',False)
        and type(block.ff.net[1]) is torch.nn.Dropout)


def make_forward(block):
    original=block._forward
    def forward(value,context=None,mask=None):
        if not compatible(block):return original(value,context=context,mask=mask)
        ge=block.ff.net[0];proj=ge.proj;final=block.ff.net[2]
        parameters=[proj.weight,*([proj.bias] if proj.bias is not None else []),
            final.weight,*([final.bias] if final.bias is not None else []),
            *[p for name in ['norm1','norm2','norm3'] for p in getattr(block,name).parameters()]]
        if (any(m.training for m in block.modules()) or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or not torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32 or not value.is_cuda
                or value.dtype!=torch.float32 or value.ndim!=3 or not value.is_contiguous()
                or value.shape[-1]!=640 or value.numel()//640!=10240
                or tuple(proj.weight.shape)!=(5120,640) or tuple(final.weight.shape)!=(640,2560)
                or (proj.bias is not None and proj.bias.shape!=(5120,))
                or (final.bias is not None and final.bias.shape!=(640,))
                or any(p.dtype!=torch.float32 or p.device!=value.device or not p.is_contiguous() for p in parameters)):
            return original(value,context=context,mask=mask)
        w=proj.weight;fw=final.weight
        try:
            key=(w.data_ptr(),w._version,tuple(w.shape),w.dtype,w.device)
            final_key=(fw.data_ptr(),fw._version,tuple(fw.shape),fw.dtype,fw.device)
        except RuntimeError:return original(value,context=context,mask=mask)
        cache=getattr(ge,'_norm3_geglu_weight_cache',None)
        final_cache=getattr(final,'_ffn640_residual_weight_cache',None)
        if torch.cuda.is_current_stream_capturing() and (
                cache is None or cache[0]!=key or final_cache is None or final_cache[0]!=final_key
                or _stream_key(value.device) not in _READY):
            return original(value,context=context,mask=mask)
        if cache is None or cache[0]!=key:
            reuse=(cache[1] if cache is not None and cache[1][0].shape==w.shape and cache[1][0].device==w.device else None)
            parts=pack_half_weight(w,out=reuse);count=_refresh_shared_ge_count(parts,refresh=True)
            if count is None:return original(value,context=context,mask=mask)
            cache=key,parts,count;ge._norm3_geglu_weight_cache=cache
        if final_cache is None or final_cache[0]!=final_key:
            reuse=(final_cache[1] if final_cache is not None and final_cache[1][0].shape==fw.shape and final_cache[1][0].device==fw.device else None)
            final_cache=final_key,pack_half_weight(fw,out=reuse)
            final._ffn640_residual_weight_cache=final_cache
        attended=block.attn1(block.norm1(value),context=context if block.disable_self_attn else None,mask=mask)
        residual,normalized=fused_residual_layer_norm(attended,value,block.norm2)
        attended=block.attn2(normalized,context=context,mask=mask)
        if (attended.shape!=residual.shape or attended.device!=residual.device or attended.dtype!=torch.float32
                or not attended.is_contiguous() or not residual.is_contiguous()):
            return original(value,context=context,mask=mask)
        residual,high,low=residual_norm_half_pack(attended,residual,block.norm3)
        hidden=primary_geglu_hidden_parts((high,low),cache[1],proj.bias)
        result=chunked_hidden_residual(hidden,final_cache[1],residual.view(-1,640),final.bias,chunk_k=512)
        if result is None:return original(value,context=context,mask=mask)
        block._decision_chunked_hidden_parts_calls=getattr(block,'_decision_chunked_hidden_parts_calls',0)+1
        return result.view_as(value)
    return forward


def install_decision_chunked_hidden_parts(diffusion):
    from unifolm_wma.modules.attention import BasicTransformerBlock
    selected=[m for m in diffusion.modules() if isinstance(m,BasicTransformerBlock) and tuple(m.norm3.normalized_shape)==(640,)]
    if any(not compatible(m) or getattr(m,'_decision_chunked_hidden_parts_installed',False) for m in selected):
        raise ValueError('Expected accepted C640 decision/world GEGLU and FFN wrappers exactly once')
    for block in selected:
        block._forward=make_forward(block);block._decision_chunked_hidden_parts_installed=True
    return len(selected)
