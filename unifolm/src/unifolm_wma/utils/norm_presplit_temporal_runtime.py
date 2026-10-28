"""Scoped reversible sequence wrapper for a diagnostic GN/SiLU/split fusion."""
import torch
from .presplit_temporal import norm_presplit_temporal_conv
from unifolm_wma.utils.triton_compensated_temporal_conv import eligible_temporal,temporal_weight_parts
from unifolm_wma.utils.triton_zero_weight_low_linear import weight_low_count


def compatible(sequence):
    if (type(sequence) is not torch.nn.Sequential or len(sequence) not in {3,4}
            or not getattr(sequence,'_split_group_norm_installed',False)
            or not isinstance(sequence[0],torch.nn.GroupNorm)
            or type(sequence[1]) is not torch.nn.SiLU or sequence[1].inplace
            or (len(sequence)==4 and type(sequence[2]) is not torch.nn.Dropout)):
        return False
    norm,conv=sequence[0],sequence[-1]
    return (norm.num_channels in {320,640} and norm.num_groups==32
        and eligible_temporal(conv) and conv.in_channels==norm.num_channels
        and conv.out_channels==norm.num_channels
        and getattr(conv,'_world_partitioned_temporal_installed',False))


class FusedNormTemporalController:
    def __init__(self,root,*,trace=False):
        self.enabled=False;self.trace=trace;self.calls=[];self.inputs={};self.cache={};self.closed=False
        self.selected=[(name,m) for name,m in root.named_modules() if compatible(m)]
        assert self.selected,'No eligible C320/C640 temporal prefix'
        assert not any(getattr(m,'_norm_presplit_temporal_installed',False) for _,m in self.selected)
        self.originals={name:m.forward for name,m in self.selected}
        for name,module in self.selected:
            module.forward=self.make_forward(name,module)
            module._norm_presplit_temporal_installed=True

    def make_forward(self,name,sequence):
        original=self.originals[name]
        def forward(value):
            if self.closed or not self.enabled or not compatible(sequence):return original(value)
            norm,conv=sequence[0],sequence[-1]
            parameters=[norm.weight,norm.bias,conv.weight,*([conv.bias] if conv.bias is not None else [])]
            if (any(m.training for m in sequence.modules()) or torch.is_grad_enabled()
                    or torch.is_autocast_enabled() or torch.backends.cudnn.allow_tf32
                    or torch.backends.cuda.matmul.allow_tf32 or not value.is_cuda
                    or value.dtype!=torch.float32 or value.ndim!=5
                    or value.shape[:3]!=(1,norm.num_channels,16)
                    or tuple(value.shape[-2:])!=((40,64) if norm.num_channels==320 else (20,32))
                    or getattr(norm,'_inference_dtype',torch.float32)!=torch.float32
                    or any(p is None or p.dtype!=torch.float32 or p.device!=value.device
                           or not p.is_contiguous() for p in parameters)):
                return original(value)
            weight=conv.weight
            try:key=(weight.data_ptr(),weight._version,tuple(weight.shape),weight.dtype,weight.device)
            except RuntimeError:return original(value)
            cache=self.cache.get(name)
            if cache is None or cache['key']!=key:
                if torch.cuda.is_current_stream_capturing():return original(value)
                reuse=cache['parts'] if cache is not None and cache['parts'][0].device==weight.device and cache['shape']==tuple(weight.shape) else None
                parts=temporal_weight_parts(weight,out=reuse)
                count=weight_low_count(parts,out=cache['count'] if reuse is not None else None)
                cache=dict(key=key,shape=tuple(weight.shape),parts=parts,count=count);self.cache[name]=cache
            if self.trace:
                self.inputs[name]=value
                self.calls.append(dict(name=name,shape=list(value.shape),strides=list(value.stride()),
                    count_ptr=cache['count'].data_ptr(),part_ptrs=[p.data_ptr() for p in cache['parts']]))
            return norm_presplit_temporal_conv(value,norm,cache['parts'],cache['count'],conv.bias)
        return forward

    def refresh_for_diagnostics(self):
        assert self.trace and set(self.inputs)==set(self.originals)
        previous=self.enabled
        try:
            for name,module in self.selected:
                self.enabled=False;self.originals[name](self.inputs[name])
                self.enabled=True;module(self.inputs[name])
        finally:self.enabled=previous

    def close(self):
        if self.closed:return
        torch.cuda.synchronize()
        for name,module in self.selected:
            module.forward=self.originals[name]
            del module._norm_presplit_temporal_installed
        self.inputs.clear();self.cache.clear();self.calls.clear();self.closed=True


def requested():
    import os
    mode=os.environ.get('UNIFOLM_WORLD_NORM_PRESPLIT_TEMPORAL','0')
    if mode not in {'0','1'}:raise ValueError('Invalid world norm/presplit temporal mode')
    if mode=='0':return False
    for key,value in [('UNIFOLM_WORLD_PARTITIONED_TEMPORAL','1'),
                      ('UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE','3'),
                      ('UNIFOLM_SPLIT_GROUP_NORM','fp32')]:
        if os.environ.get(key)!=value:raise ValueError('Norm/presplit temporal requires '+key+'='+value)
    return True
