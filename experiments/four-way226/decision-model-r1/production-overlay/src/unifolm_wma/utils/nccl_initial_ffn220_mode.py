"""Shard the omitted world init_attn FFN on the actual adopted219 baseline."""
from contextlib import contextmanager
import os
from pathlib import Path
import torch
import torch.distributed as dist
from . import nccl_native_attention219_mode as adopted
from unifolm_wma.utils.initial_native_hidden_parts_pipeline import _bk32_native_hidden_parts as PRODUCER
from unifolm_wma.utils.triton_prepacked_input_linear import _prepacked_gemm as CONSUMER

enabled=False
active=None


def row_values(values,kind,rank):
    """Route only tensor rows and M; all K arithmetic and tile options survive."""
    assert rank in (0,1,2,3) and values['M']==40960
    m=values['M'];part=m//4;rows=slice(rank*part,(rank+1)*part)
    result=dict(values)
    if kind=='producer':
        assert (values['C'],values['K'],values['BM'],values['BN'],values['GROUP'])==(2048,512,64,64,32)
        assert not values['DEBUG']
        buffers=(('X',512),('YH',2048),('YL',2048))
        n=values['C']
    else:
        assert kind=='consumer'
        assert (values['N'],values['K'],values['BM'],values['BN'],values['BK'],values['GROUP_M'],values['OUTPUT_GROUPS'])==(512,2048,64,128,32,32,1)
        buffers=(('XH',2048),('XL',2048),('Output',512))
        n=values['N']
    # Both halves begin/end at original complete GROUP*BM boundaries.
    group=values['GROUP'] if kind=='producer' else values['GROUP_M']
    assert part%(group*values['BM'])==0
    for name,width in buffers:
        tensor=values[name]
        assert tensor.is_contiguous() and tensor.numel()==m*width
        result[name]=tensor.view(m,width)[rows]
    result['M']=part
    grid=((part+values['BM']-1)//values['BM']*((n+values['BN']-1)//values['BN']),)
    return result,grid


@contextmanager
def new_shape_cache(rank):
    # Isolate only previously unused M=20480 variants; never replace the cache
    # manager or change the cache used by original full-shape kernel checks.
    assert 'TRITON_CACHE_MANAGER' not in os.environ
    previous=os.environ.get('TRITON_CACHE_DIR')
    root=Path(os.environ['INIT_FFN_CACHE_DIR']);assert root.is_dir()
    directory=root/('rank-'+str(rank));directory.mkdir(mode=0o700,exist_ok=True)
    os.environ['TRITON_CACHE_DIR']=str(directory)
    try:yield
    finally:
        if previous is None:os.environ.pop('TRITON_CACHE_DIR',None)
        else:os.environ['TRITON_CACHE_DIR']=previous
        assert os.environ.get('TRITON_CACHE_DIR')==previous


class InitialFFNController:
    def __init__(self,rank=None,gather=None):
        if rank is None:
            assert dist.is_initialized() and dist.get_world_size()==4
            rank=dist.get_rank()
        assert rank in (0,1,2,3)
        self.rank=rank;self.gather=dist.all_gather_into_tensor if gather is None else gather
        self.pending={};self.captures=[];self.runs={}
        for jit in (PRODUCER,CONSUMER):
            self.runs[jit]=jit.run
            def launch(*args,grid=None,warmup=False,_jit=jit,**kwargs):
                if warmup:return self.runs[_jit](*args,grid=grid,warmup=True,**kwargs)
                return self.launch(_jit,args,grid,kwargs)
            jit.run=launch
    def call(self,jit,values,grid,kwargs):
        options={k:v for k,v in kwargs.items() if k not in jit.arg_names}
        with new_shape_cache(self.rank):
            return self.runs[jit](*[values[n] for n in jit.arg_names],grid=grid,warmup=False,**options)
    def launch(self,jit,args,grid,kwargs):
        old=self.runs[jit]
        if not enabled or torch.backends.cudnn.allow_tf32:return old(*args,grid=grid,warmup=False,**kwargs)
        values=dict(zip(jit.arg_names,args));values.update(kwargs)
        if jit is CONSUMER:
            key=tuple(values[n].data_ptr() for n in ('XH','XL'))
            if key not in self.pending:return old(*args,grid=grid,warmup=False,**kwargs)
        assert set(jit.arg_names)<=values.keys()
        if jit is PRODUCER:
            assert not self.pending
            assert (values['M'],values['C'],values['K'])==(40960,2048,512)
            assert not torch.is_grad_enabled() and not torch.backends.cuda.matmul.allow_tf32
            assert values['X'].is_cuda and values['X'].dtype==torch.float32
            assert kwargs['num_warps']==4 and kwargs['num_stages']==2 and kwargs['enable_fp_fusion'] is False
            key=tuple(values[n].data_ptr() for n in ('YH','YL'))
            self.pending[key]=dict(M=values['M'],C=values['C'],K=values['K'])
            routed,new_grid=row_values(values,'producer',self.rank)
            return self.call(jit,routed,new_grid,kwargs)
        key=tuple(values[n].data_ptr() for n in ('XH','XL'))
        pending=self.pending.pop(key,None)
        if pending is None:return old(*args,grid=grid,warmup=False,**kwargs)
        assert (values['M'],values['N'],values['K'])==(pending['M'],pending['K'],pending['C'])
        assert kwargs['num_warps']==4 and kwargs['num_stages']==3 and kwargs['enable_fp_fusion'] is False
        output=values['Output'];routed,new_grid=row_values(values,'consumer',self.rank)
        result=self.call(jit,routed,new_grid,kwargs)
        self.gather(output.view(40960,512),routed['Output'])
        if torch.cuda.is_current_stream_capturing():
            self.captures.append(dict(shape=[40960,512,2048],rank=self.rank,collectives=1,
                original_producer_and_consumer_JIT_bodies=True,original_BK32=True,
                original_group_alignment=True,outer_full_shape_retained=True,
                output_half_MiB=routed['Output'].numel()*routed['Output'].element_size()/2**20))
        return result
    def close(self):
        assert not self.pending
        for jit,run in self.runs.items():jit.run=run
    def report(self):
        assert not self.pending
        return dict(captured=len(self.captures),rows=self.captures.copy(),
            original_producer_and_consumer_JIT_bodies=True,original_BK32=True,
            original_master_version_cache_retained=True,one_gather_per_whole_initial_FFN=True,
            no_activation_or_output_replay_cache=True,default_old_kernel_cache_restored=True)


class Controller:
    def __init__(self,model):
        self.base=adopted.install(model)
        self.initial=InitialFFNController()
    @property
    def captures(self):return self.base.captures
    def report(self):
        report=self.base.report();report['initial_FFN']=self.initial.report()
        return report


def install(model):
    global active
    assert active is None;active=Controller(model);return active


def uninstall():
    global active,enabled
    if active is not None:
        active.initial.close();adopted.uninstall()
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:
        with adopted.mode(value):yield
    finally:
        assert not active.initial.pending
        enabled=previous
