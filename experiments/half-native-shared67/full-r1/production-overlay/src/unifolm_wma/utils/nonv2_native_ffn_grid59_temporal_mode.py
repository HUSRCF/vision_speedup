"""Share existing FFN gathers with native temporal Half self-attention.

The full 5D GroupNorm, all original GEMM descriptors, and all 16 time tokens
remain unchanged. Only independent spatial-position batches are partitioned at
the ordinary Torch/Triton tensor boundary. No new native argument inspection.
"""
import hashlib
import torch
import torch.distributed as dist
import triton
import xformers.ops
from . import nonv2_native_ffn_grid59_spatial_mode as spatial
from unifolm_wma.utils.triton_temporal_attention import _temporal_attention

geglu=spatial.geglu
base=spatial.base
active=None


def local_temporal_arguments(query,key,value,output,rank):
    assert rank in (0,1) and query.ndim==3 and query.shape==key.shape==value.shape==output.shape
    assert query.shape[0]%2==0 and query.shape[1]==16
    assert all(t.is_contiguous() for t in (query,key,value,output))
    half=query.shape[0]//2
    return tuple(t[rank*half:(rank+1)*half] for t in (query,key,value,output))


def launch(jit,*args,**kwargs):
    if jit is not _temporal_attention or active is None or kwargs.get('warmup',False):return geglu.launch(jit,*args,**kwargs)
    q,k,v,out,qk,prob,heads,scale,accurate,debug=args
    key=(heads,scale,accurate,debug)
    if not base.enabled or active.temporal_scope is None:
        result=geglu.launch(jit,*args,**kwargs)
        digest=hashlib.sha256(result.asm['cubin']).hexdigest()
        previous=active.temporal_cubins.setdefault(key,digest);assert previous==digest
        return result
    assert len(args)==10 and not debug and q.shape[0]==active.temporal_scope['batch']
    assert tuple(kwargs['grid'])==(q.shape[0]*heads,) and key in active.temporal_cubins
    a,b,c,y=local_temporal_arguments(q,k,v,out,active.rank)
    result=geglu.launch(jit,a,b,c,y,y,y,heads,scale,accurate,debug,**dict(kwargs,grid=(a.shape[0]*heads,)))
    digest=hashlib.sha256(result.asm['cubin']).hexdigest();assert digest==active.temporal_cubins[key]
    if torch.cuda.is_current_stream_capturing():
        active.temporal_calls.append(dict(module=active.temporal_scope['name'],query_shape=list(q.shape),local_batch=a.shape[0],full16_time_tokens=True,no_new_collective=True,original_CUBIN_SHA256=digest,original_constexpr_and_rounding=True))
    return result


def local_temporal_attention(parent,q,k,v,bias,rank,**options):
    assert rank in (0,1) and q.ndim==k.ndim==v.ndim==4
    batch=q.shape[0]
    assert batch%2==0 and k.shape[0]==v.shape[0]==batch
    assert q.shape[1]==k.shape[1]==v.shape[1]==16
    assert options.get('p',0.)==0.
    start,stop=rank*(batch//2),(rank+1)*(batch//2)
    if isinstance(bias,torch.Tensor):
        assert bias.ndim==4 and bias.shape[0] in (1,batch)
        bias=bias[start:stop] if bias.shape[0]==batch else bias
    else:assert bias is None
    result=parent(q[start:stop],k[start:stop],v[start:stop],attn_bias=bias,**options)
    assert result.shape==(batch//2,16,q.shape[2],v.shape[3])
    output=torch.empty((batch,*result.shape[1:]),device=result.device,dtype=result.dtype)
    output[start:stop].copy_(result)
    return output


def gather_rows(value,rank,gather):
    if active is not None and active.temporal_scope is not None:
        assert rank==active.rank and value.shape[:2]==(active.temporal_scope['batch'],16)
        return value
    return spatial.gather_rows(value,rank,gather)


def attention(q,k,v,attn_bias=None,p=0.,scale=None,*,op=None,output_dtype=None):
    if not base.enabled or active is None or active.temporal_scope is None:
        return spatial.attention(q,k,v,attn_bias=attn_bias,p=p,scale=scale,op=op,output_dtype=output_dtype)
    assert active.spatial_scope is None and q.shape[0]==active.temporal_scope['batch']
    assert not torch.is_grad_enabled()
    assert all(t.is_cuda and t.dtype==torch.float16 and t.device==q.device for t in (q,k,v))
    result=local_temporal_attention(spatial.parent_attention,q,k,v,attn_bias,active.rank,p=p,scale=scale,op=op,output_dtype=output_dtype)
    if torch.cuda.is_current_stream_capturing():
        active.temporal_calls.append(dict(module=active.temporal_scope['name'],query_shape=list(q.shape),key_shape=list(k.shape),local_batch=q.shape[0]//2,full16_time_tokens=True,no_new_collective=True))
    return result


class Controller(spatial.Controller):
    def __init__(self,model):
        from unifolm_wma.modules.attention import TemporalTransformer
        self.temporal_scope=None;self.temporal_rows=[];self.temporal_calls=[];self.temporal_modules=[];self.temporal_cubins={}
        super().__init__(model)
        for name,module in model.named_modules():
            if type(module)is not TemporalTransformer:continue
            assert module.only_self_att and len(module.transformer_blocks)==1
            self.temporal_modules.append(module)
            block=module.transformer_blocks[0];before=block.forward
            def forward(value,*args,_before=before,_name=name,**kwargs):
                if not base.enabled or not base.choose(value.shape,self.qualified):return _before(value,*args,**kwargs)
                assert self.temporal_scope is None and self.spatial_scope is None
                assert value.ndim==3 and value.shape[0]%2==0 and value.shape[1]==16
                assert value.is_cuda and value.dtype==torch.float16 and value.is_contiguous()
                self.temporal_scope=dict(name=_name,batch=value.shape[0])
                try:
                    result=_before(value,*args,**kwargs)
                    assert result.shape==value.shape and result.is_contiguous()
                    full=spatial.parent_gather(result,self.rank,dist.all_gather_into_tensor)
                    if torch.cuda.is_current_stream_capturing():
                        self.temporal_rows.append(dict(module=_name,input_shape=list(value.shape),collectives=1,moved_existing_FFN_gather=True,full5D_GroupNorm_and_full16_time_tokens=True,full_vendor_shapes_preserved=True))
                    return full
                finally:self.temporal_scope=None
            block.forward=forward;self.saved.append((block,before))
        assert len(self.temporal_modules)==17

    def masters(self):
        result=super().masters()
        result.update({p.data_ptr():p for module in self.temporal_modules for p in module.parameters()})
        return result

    def report(self):
        report=super().report()
        report['rows']=[dict(row,collectives=0) for row in report['rows']]
        report['temporal_shared_boundary']=dict(captured=len(self.temporal_rows),rows=self.temporal_rows.copy(),attention_calls=len(self.temporal_calls),calls=self.temporal_calls.copy(),new_collectives=0,FFN_gather_moved_to_end_of_temporal_inner_block=True,GroupNorm_before_partition_full_input=True)
        return report


def install(model):
    global active
    assert active is None and spatial.active is None and geglu.active is None and base.active is None
    assert triton.runtime.jit.JITFunction.run is geglu.parent_run
    assert xformers.ops.memory_efficient_attention is spatial.parent_attention and base.gather_rows is spatial.parent_gather
    active=Controller(model);spatial.active=active;geglu.active=active;base.active=active
    triton.runtime.jit.JITFunction.run=launch
    base.gather_rows=gather_rows;xformers.ops.memory_efficient_attention=attention
    return active


def uninstall():
    global active
    assert base.gather_rows is gather_rows and xformers.ops.memory_efficient_attention is attention
    assert triton.runtime.jit.JITFunction.run is launch
    triton.runtime.jit.JITFunction.run=geglu.launch
    base.gather_rows=spatial.parent_gather;xformers.ops.memory_efficient_attention=spatial.parent_attention
    geglu.uninstall();spatial.active=None;active=None


mode=geglu.mode
