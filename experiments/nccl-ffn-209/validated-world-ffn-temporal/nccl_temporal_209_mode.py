"""Shard independent temporal-convolution output channels with original math."""
import torch
import torch.distributed as dist
import nccl_ffn_209_mode as parent
from nccl_ffn_209_mode import measure,mode
from unifolm_wma.utils.presplit_temporal import _presplit_temporal as PRESPLIT
from unifolm_wma.utils.triton_zero_weight_low_temporal import _zero_low_temporal as ZERO

class Controller(parent.Controller):
    def __init__(self,model):
        super().__init__(model);self.temporal_captures=[];self.temporal_pending={}
        for jit in (PRESPLIT,ZERO):
            self.runs[jit]=jit.run
            def launch(*args,grid=None,warmup=False,_jit=jit,**kwargs):
                if warmup:return self.runs[_jit](*args,grid=grid,warmup=True,**kwargs)
                return self.temporal(_jit,args,grid,kwargs)
            jit.run=launch

    def temporal(self,jit,args,grid,kwargs):
        old=self.runs[jit]
        if not parent.enabled:return old(*args,grid=grid,warmup=False,**kwargs)
        values=dict(zip(jit.arg_names,args));values.update(kwargs)
        assert set(jit.arg_names)<=values.keys()
        m,n,c,t,s=(values[k] for k in ('M','N','C','T','S'))
        if (m!=t*s or t!=16 or n!=c or c not in (320,640,1280) or torch.backends.cudnn.allow_tf32):
            return old(*args,grid=grid,warmup=False,**kwargs)
        assert not torch.backends.cuda.matmul.allow_tf32 and kwargs['enable_fp_fusion'] is False
        part=n//2;channels=slice(self.rank*part,(self.rank+1)*part)
        name='Output' if jit is PRESPLIT else 'Y';output=values[name]
        assert output.dtype==torch.float32 and output.is_contiguous() and output.numel()==m*n
        for slot in ('WH','WL'):
            value=values[slot];assert value.is_contiguous() and value.shape==(n,c*3)
            values[slot]=value[channels]
        if values['HAS_BIAS']:
            assert values['Bias'].shape==(n,);values['Bias']=values['Bias'][channels]
        values[name]=output.view(n,m)[channels];values['N']=part
        bm,bn=(values['BM'],values['BN']) if jit is PRESPLIT else (64,64)
        options={k:v for k,v in kwargs.items() if k not in jit.arg_names}
        result=old(*[values[k] for k in jit.arg_names],grid=((m+bm-1)//bm*((part+bn-1)//bn),),warmup=False,**options)
        key=output.data_ptr()
        if jit is PRESPLIT and not values['THREE']:
            assert key not in self.temporal_pending;self.temporal_pending[key]=[m,n,c,t,s]
            return result
        if jit is PRESPLIT:assert self.temporal_pending.pop(key)==[m,n,c,t,s]
        target=output.view(n,m);dist.all_gather_into_tensor(target,target[channels])
        if torch.cuda.is_current_stream_capturing():self.temporal_captures.append(dict(shader=jit.fn.__name__,geometry=[m,n,c,t,s],rank=self.rank,output_channel_shard=True,original_full_input_C_and_K_order=True,original_full_weight_Count=True,collectives=1))
        return result

    def report(self):
        assert not self.temporal_pending
        result=super().report();result['temporal_captured']=len(self.temporal_captures);result['temporal_rows']=self.temporal_captures.copy()
        return result

def install(model):
    assert parent.active is None
    parent.active=Controller(model);return parent.active

def uninstall():
    if parent.active is not None:assert not parent.active.temporal_pending
    parent.uninstall()
