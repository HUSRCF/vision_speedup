"""Two original FFN shaders per rank, one final collective per whole FFN."""
from contextlib import contextmanager
import torch
import torch.distributed as dist
from unifolm_wma.utils.triton_geglu_subtile32 import _geglu_subtile32 as GEGLU
from unifolm_wma.utils.triton_geglu_hidden_parts_kernels import _packed_two_product_gemm_residual as FINAL640
from unifolm_wma.utils.ffn1280_zero_residual import _zero_low_gemm_residual as FINAL320
from nccl_geglu_209_mode import measure

enabled=False
active=None

class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import BasicTransformerBlock
        assert dist.get_world_size()==2
        self.rank=dist.get_rank();self.pending={};self.captures=[];self.saved=[];self.runs={}
        for jit in (GEGLU,FINAL640,FINAL320):
            self.runs[jit]=jit.run
            def launch(*args,grid=None,warmup=False,_jit=jit,**kwargs):
                if warmup:return self.runs[_jit](*args,grid=grid,warmup=True,**kwargs)
                return self.launch(_jit,args,grid,kwargs)
            jit.run=launch
        for module in model.modules():
            if not isinstance(module,BasicTransformerBlock):continue
            original=module._forward
            def forward(*args,_original=original,**kwargs):
                assert not self.pending
                output=_original(*args,**kwargs)
                assert not self.pending,'Sharded hidden must be consumed and gathered within its original block'
                return output
            module._forward=forward;self.saved.append((module,original))

    def call(self,jit,values,kwargs):
        m=values['M'];n=values.get('C',values.get('N'));bm=values['BM'];bn=values.get('BN',64)
        options={n:v for n,v in kwargs.items() if n not in jit.arg_names}
        return self.runs[jit](*[values[n] for n in jit.arg_names],grid=((m+bm-1)//bm*((n+bn-1)//bn),),warmup=False,**options)

    def launch(self,jit,args,grid,kwargs):
        old=self.runs[jit]
        if not enabled:return old(*args,grid=grid,warmup=False,**kwargs)
        values=dict(zip(jit.arg_names,args));values.update(kwargs)
        assert set(jit.arg_names)<=values.keys()
        m,k=values['M'],values['K'];part=m//2;rows=slice(self.rank*part,(self.rank+1)*part)
        if jit is GEGLU:
            c=values['C']
            if (m,c,k) not in ((10240,2560,640),(40960,1280,320)):
                return old(*args,grid=grid,warmup=False,**kwargs)
            assert not self.pending and not values['DEBUG'] and not torch.backends.cudnn.allow_tf32
            assert values['BK']==64 and values['PRIMARY_GATE'] and not values['PRIMARY_VALUE']
            key=tuple(values[n].data_ptr() for n in ('YH','YL'));self.pending[key]=dict(M=m,K=k,C=c)
            for name,width in (('XH',k),('XL',k),('YH',c),('YL',c)):
                value=values[name];assert value.is_contiguous() and value.numel()==m*width
                values[name]=value.view(m,width)[rows]
            values['M']=part
            return self.call(jit,values,kwargs)
        key=tuple(values[n].data_ptr() for n in ('XH','XL'))
        pending=self.pending.pop(key,None)
        if pending is None:return old(*args,grid=grid,warmup=False,**kwargs)
        n=values['N'];assert (pending['M'],pending['C'],pending['K'])==(m,k,n)
        assert kwargs['enable_fp_fusion'] is False and not torch.is_grad_enabled()
        if jit is FINAL320:assert values['OUTPUT_GROUPS']==1
        name='Output' if jit is FINAL640 else 'Y';output=values[name]
        for slot,width in (('XH',k),('XL',k),('Residual',n),(name,n)):
            value=values[slot];assert value.is_contiguous() and value.numel()==m*width
            values[slot]=value.view(m,width)[rows]
        values['M']=part;result=self.call(jit,values,kwargs)
        target=output.view(m,n);dist.all_gather_into_tensor(target,target[rows])
        if torch.cuda.is_current_stream_capturing():self.captures.append(dict(shape=[m,n,k],rank=self.rank,collectives=1,unchanged_two_shader_bodies=True))
        return result

    def report(self):
        assert not self.pending
        return dict(captured=len(self.captures),rows=self.captures.copy(),unchanged_shader_bodies=True,original_K64_order=True,
            original_full_shape_owners=True,one_final_output_allgather_per_FFN=True,two_GPUs=True)

def install(model):
    global active
    assert active is None;active=Controller(model);return active

def uninstall():
    global active,enabled
    if active is not None:
        assert not active.pending
        for jit,run in active.runs.items():jit.run=run
        for module,forward in active.saved:module._forward=forward
    active=None;enabled=False

@contextmanager
def mode(value):
    global enabled
    prior=enabled;enabled=value
    try:yield
    finally:assert not active.pending;enabled=prior
