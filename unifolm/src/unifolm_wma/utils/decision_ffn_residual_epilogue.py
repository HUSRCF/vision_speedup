"""Decision C1280 FFN residual fusion, preserving existing block arithmetic."""
import torch
from .triton_residual_layer_norm import fused_residual_layer_norm
from .triton_half_compensated_linear import pack_half_weight
from .cublas_compensated_linear import _READY,_stream_key
from .decision_primary_residual import primary_linear_residual


def compatible(block):
    from unifolm_wma.modules.attention import FeedForward,GEGLU
    if (not getattr(block,'_inference_fused_residual_norm',False)
            or not isinstance(block.ff,FeedForward) or block.ff._inference_dtype!=torch.float32
            or len(block.ff.net)!=3 or not isinstance(block.ff.net[0],GEGLU)
            or not isinstance(block.ff.net[1],torch.nn.Dropout)
            or not isinstance(block.ff.net[2],torch.nn.Linear)
            or any(not isinstance(getattr(block,n),torch.nn.LayerNorm)
                   or tuple(getattr(block,n).normalized_shape)!=(1280,)
                   for n in ['norm1','norm2','norm3'])):
        return False
    ge=block.ff.net[0].proj;final=block.ff.net[2]
    return (ge.in_features==1280 and ge.out_features==10240
        and final.in_features==5120 and final.out_features==1280
        and all(getattr(final,key,False) for key in ['_tf32x3_installed',
            '_decision_vendor_ffn_installed','_decision_primary_ffn_installed']))


def make_forward(block):
    original=block._forward
    def forward(value,context=None,mask=None):
        if not compatible(block):return original(value,context=context,mask=mask)
        final=block.ff.net[2];ge=block.ff.net[0].proj
        parameters=[final.weight,*([final.bias] if final.bias is not None else []),
            ge.weight,*([ge.bias] if ge.bias is not None else []),
            *[t for name in ['norm1','norm2','norm3'] for t in getattr(block,name).parameters()]]
        if (any(m.training for m in block.modules()) or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or not torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32 or not value.is_cuda
                or value.dtype!=torch.float32 or value.ndim!=3 or not value.is_contiguous()
                or value.shape[-1]!=1280 or value.numel()//1280 not in {2560,640}
                or any(t.dtype!=torch.float32 or t.device!=value.device
                       or not t.is_contiguous() for t in parameters)):
            return original(value,context=context,mask=mask)
        w=final.weight
        try:key=(w.data_ptr(),w._version,tuple(w.shape),w.dtype,w.device)
        except RuntimeError:return original(value,context=context,mask=mask)
        cache=getattr(final,'_decision_ffn_residual_weight_cache',None)
        capture=torch.cuda.is_current_stream_capturing()
        if capture and (cache is None or cache[0]!=key or _stream_key(value.device) not in _READY):
            return original(value,context=context,mask=mask)
        if cache is None or cache[0]!=key:
            reuse=(cache[1] if cache is not None and cache[1][0].shape==w.shape
                   and cache[1][0].device==w.device else None)
            cache=key,pack_half_weight(w,out=reuse)
            final._decision_ffn_residual_weight_cache=cache
        attended=block.attn1(block.norm1(value),
            context=context if block.disable_self_attn else None,mask=mask)
        residual,normalized=fused_residual_layer_norm(attended,value,block.norm2)
        attended=block.attn2(normalized,context=context,mask=mask)
        residual,normalized=fused_residual_layer_norm(attended,residual,block.norm3)
        hidden=block.ff.net[1](block.ff.net[0](normalized))
        if (hidden.dtype!=torch.float32 or not hidden.is_contiguous()
                or not residual.is_contiguous() or hidden.shape!=(*value.shape[:-1],5120)
                or residual.shape!=value.shape):
            return final(hidden)+residual
        result=primary_linear_residual(hidden,cache[1][0],residual,final.bias)
        if result is None:return final(hidden)+residual
        block._decision_ffn_residual_epilogue_calls=getattr(block,'_decision_ffn_residual_epilogue_calls',0)+1
        return result
    return forward


def install_decision_ffn_residual_epilogue(diffusion):
    from unifolm_wma.modules.attention import BasicTransformerBlock
    selected=[m for m in diffusion.modules() if isinstance(m,BasicTransformerBlock)
              and tuple(m.norm3.normalized_shape)==(1280,)]
    if any(not compatible(m) or getattr(m,'_decision_ffn_residual_epilogue_installed',False)
           for m in selected):
        raise ValueError('Expected original C1280 decision primary FFN and residual norm')
    for block in selected:
        block._forward=make_forward(block)
        block._decision_ffn_residual_epilogue_installed=True
    return len(selected)
