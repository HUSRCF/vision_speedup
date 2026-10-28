"""Shard only independent GEGLU rows; retain original public tensor shapes."""
from contextlib import contextmanager
import torch
import torch.distributed as dist
from unifolm_wma.utils.triton_geglu_subtile32 import _geglu_subtile32 as JIT

enabled=False
active=None
original_run=None

def measure(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype==torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    exact=torch.equal(actual.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))
    error=dict(relative_l2=float((actual-expected).norm()/expected.norm().clamp_min(1e-30)),max_abs=float((actual-expected).abs().max()),signedzero_bits=exact)
    assert exact,error
    return error

class Controller:
    def __init__(self):
        assert dist.get_world_size()==2
        self.rank=dist.get_rank();self.captures=[]

    def launch(self,args,grid,kwargs):
        if not enabled:return original_run(*args,grid=grid,warmup=False,**kwargs)
        values=dict(zip(JIT.arg_names,args));values.update(kwargs)
        assert set(JIT.arg_names)<=values.keys()
        m,c,k=(values[n] for n in ('M','C','K'))
        if (m,c,k)!=(10240,2560,640):return original_run(*args,grid=grid,warmup=False,**kwargs)
        assert not values['DEBUG']
        assert values['BM'] in (32,64) and values['BN']==64 and values['BK']==64
        assert values['PRIMARY_VALUE'] is False and values['PRIMARY_GATE'] is True
        assert kwargs['enable_fp_fusion'] is False and not torch.is_grad_enabled()
        part=m//2;rows=slice(self.rank*part,(self.rank+1)*part)
        for name,width in (('XH',k),('XL',k),('YH',c),('YL',c)):
            value=values[name];assert value.is_contiguous() and value.numel()==m*width
            values[name]=value.view(m,width)[rows]
        values['M']=part
        arguments=[values[n] for n in JIT.arg_names]
        launch_options={n:v for n,v in kwargs.items() if n not in JIT.arg_names}
        bm,bn=values['BM'],values['BN']
        result=original_run(*arguments,grid=((part+bm-1)//bm*((c+bn-1)//bn),),warmup=False,**launch_options)
        for name in ('YH','YL'):
            full=dict(zip(JIT.arg_names,args));full.update(kwargs)
            target=full[name].view(m,c)
            dist.all_gather_into_tensor(target,target[rows])
        if torch.cuda.is_current_stream_capturing():self.captures.append(dict(shape=[m,c,k],rank=self.rank,rows=[self.rank*part,(self.rank+1)*part]))
        return result

    def report(self):
        return dict(captured=len(self.captures),rows=self.captures.copy(),unchanged_shader=True,original_K64_order=True,original_full_shape_owners=True,output_allgather_inside_graph=True,two_GPUs=True)

def launched(*args,grid=None,warmup=False,**kwargs):
    if warmup:return original_run(*args,grid=grid,warmup=True,**kwargs)
    return active.launch(args,grid,kwargs)

def install(configs):
    global active,original_run
    assert active is None
    original_run=JIT.run;active=Controller();JIT.run=launched
    return active

def uninstall():
    global active,original_run,enabled
    if active is not None:assert JIT.run is launched;JIT.run=original_run
    active=None;original_run=None;enabled=False

@contextmanager
def mode(value):
    global enabled
    prior=enabled;enabled=value
    try:yield
    finally:enabled=prior
