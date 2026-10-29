"""Share qualified FFN gathers with native spatial Half attention."""
import torch
import torch.distributed as dist
import triton
import xformers.ops
from . import nonv2_native_ffn_grid59_geglu_mode as geglu
from .nonv2_native_attention59_mode import local_attention

base=geglu.adopted
parent_attention=xformers.ops.memory_efficient_attention
parent_gather=base.gather_rows
active=None

def gather_rows(value,rank,gather):
    if active is not None and active.spatial_scope is not None:
        assert rank==active.rank and value.shape[0]==16
        return value
    return parent_gather(value,rank,gather)

def attention(q,k,v,attn_bias=None,p=0.,scale=None,*,op=None,output_dtype=None):
    options=dict(p=p,scale=scale,op=op,output_dtype=output_dtype)
    if not base.enabled or active is None or active.spatial_scope is None:
        return parent_attention(q,k,v,attn_bias=attn_bias,**options)
    scope=active.spatial_scope
    assert not torch.is_grad_enabled() and q.shape[1]==scope['M']
    assert all(t.is_cuda and t.dtype==torch.float16 and t.device==q.device for t in (q,k,v))
    result=local_attention(parent_attention,q,k,v,attn_bias,active.rank,**options)
    if torch.cuda.is_current_stream_capturing():
        active.spatial_calls.append(dict(module=scope['name'],query_shape=list(q.shape),key_shape=list(k.shape),local_batch=8,original_global_mask_slice=True,no_new_collective=True))
    return result

class Controller(geglu.ProductionController):
    def __init__(self,model):
        from unifolm_wma.modules.attention import SpatialTransformer
        self.spatial_scope=None;self.spatial_rows=[];self.spatial_calls=[];self.spatial_modules=[]
        super().__init__(model)
        for name,module in model.named_modules():
            if type(module)is not SpatialTransformer:continue
            self.spatial_modules.append(module);before=module.forward
            assert len(module.transformer_blocks)==1 and not module._channels_last_output
            def forward(value,*args,_before=before,_name=name,**kwargs):
                shape=(value.shape[0],value.shape[2]*value.shape[3],value.shape[1])
                if not base.enabled or not base.choose(shape,self.qualified):return _before(value,*args,**kwargs)
                assert self.spatial_scope is None and value.shape[0]==16 and value.dtype==torch.float16
                self.spatial_scope=dict(name=_name,M=shape[1])
                try:
                    result=_before(value,*args,**kwargs)
                    assert result.shape==value.shape and result.is_contiguous()
                    full=torch.empty_like(result,memory_format=torch.contiguous_format)
                    dist.all_gather_into_tensor(full,result[self.rank*8:(self.rank+1)*8].contiguous())
                    if torch.cuda.is_current_stream_capturing():
                        self.spatial_rows.append(dict(module=_name,input_shape=list(value.shape),collectives=1,moved_existing_FFN_gather=True,full_vendor_shapes_and_GroupNorm_preserved=True))
                    return full
                finally:self.spatial_scope=None
            module.forward=forward;self.saved.append((module,before))
        assert len(self.spatial_modules)==16
    def masters(self):
        result=super().masters()
        result.update({p.data_ptr():p for module in self.spatial_modules for p in module.parameters()})
        return result
    def report(self):
        report=super().report()
        report['rows']=[dict(row,collectives=0 if row['input_shape'][0]==16 else 1)for row in report['rows']]
        report['spatial_shared_boundary']=dict(captured=len(self.spatial_rows),rows=self.spatial_rows.copy(),attention_calls=len(self.spatial_calls),calls=self.spatial_calls.copy(),new_collectives=0,FFN_gather_moved_to_end_of_SpatialTransformer=True)
        return report

def install(model):
    global active
    assert active is None and geglu.active is None and base.active is None
    assert triton.runtime.jit.JITFunction.run is geglu.parent_run
    assert xformers.ops.memory_efficient_attention is parent_attention and base.gather_rows is parent_gather
    active=Controller(model);geglu.active=active;base.active=active
    triton.runtime.jit.JITFunction.run=geglu.launch
    base.gather_rows=gather_rows;xformers.ops.memory_efficient_attention=attention
    return active

def uninstall():
    global active
    assert base.gather_rows is gather_rows and xformers.ops.memory_efficient_attention is attention
    base.gather_rows=parent_gather;xformers.ops.memory_efficient_attention=parent_attention
    geglu.uninstall();active=None

mode=geglu.mode
