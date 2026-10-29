"""Exact per-stage, per-timestep memoization of explicitly static MLP outputs.

Only register the WMA time/fps embeddings and ResBlock.emb_layers. Dynamic
video/action/state features, attention and sampler updates are never cached.
All allocation/initial filling remains in the original process timer.
"""
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import torch


def tensor_key(tensor):
    assert isinstance(tensor,torch.Tensor)and tensor.numel()>0
    # Read only tiny declared sampler timesteps/fs, never a feature or weight.
    return tuple(tensor.shape),str(tensor.dtype),tuple(tensor.detach().cpu().reshape(-1).tolist())


# These kernels only copy typed values. All addresses are ordinary Tensor
# arguments plus the declared slot index, with no native graph introspection.
import triton
import triton.language as tl

@triton.jit
def _store_slot(Value,Table,Index,N:tl.constexpr,BLOCK:tl.constexpr,CACHE_REVISION:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    slot=tl.load(Index)
    value=tl.load(Value+offsets,offsets<N,other=0)
    tl.store(Table+slot*N+offsets,value,offsets<N)

@triton.jit
def _load_slot(Table,Value,Index,N:tl.constexpr,BLOCK:tl.constexpr,CACHE_REVISION:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    slot=tl.load(Index)
    value=tl.load(Table+slot*N+offsets,offsets<N,other=0)
    tl.store(Value+offsets,value,offsets<N)



@contextmanager
def kernel_cache_scope():
    import triton.runtime.cache as cache
    rank=int(os.environ['TEAM07_PARALLEL_RANK']);assert rank in(0,1)
    root=Path(os.environ['TEAM07_STATIC_TIME_CACHE'])/('rank'+str(rank))
    assert(root/'gpu').is_dir()and(root/'exec').is_dir()
    names=('TRITON_CACHE_DIR','TRITON_CACHE_MANAGER','EXPERIMENT_EXEC_CACHE')
    previous={name:os.environ.get(name)for name in names}
    previous_class=cache.__dict__['__cache_cls'];previous_name=cache.__dict__['__cache_cls_nme']
    os.environ.update(TRITON_CACHE_DIR=str(root/'gpu'),EXPERIMENT_EXEC_CACHE=str(root/'exec'),TRITON_CACHE_MANAGER='split_experiment_triton_cache:ReadWarmSplitCache')
    try:yield
    finally:
        cache.__dict__['__cache_cls']=previous_class;cache.__dict__['__cache_cls_nme']=previous_name
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def store_slot(table,index,value):
    if value.is_cuda:
        with kernel_cache_scope():_store_slot[(triton.cdiv(value.numel(),256),)](value,table,index,value.numel(),BLOCK=256,CACHE_REVISION=4)
    else:table.index_copy_(0,index,value.unsqueeze(0))


def load_slot(table,index):
    if not table.is_cuda:return table.index_select(0,index).squeeze(0)
    value=torch.empty(table.shape[1:],device=table.device,dtype=table.dtype)
    with kernel_cache_scope():_load_slot[(triton.cdiv(value.numel(),256),)](table,value,index,value.numel(),BLOCK=256,CACHE_REVISION=4)
    return value


@torch.no_grad()
def graph_copy_selfcheck(device):
    assert torch.device(device).type=='cuda'
    checked=0
    for dtype,integer in((torch.float16,torch.int16),(torch.float32,torch.int32)):
        index=torch.zeros(1,dtype=torch.long,device=device)
        value=torch.arange(17,device=device,dtype=dtype)
        value[0]=-0.;value[1]=float('inf');value[2]=float('-inf');value[3]=float('nan')
        table=torch.empty((50,17),device=device,dtype=dtype)
        current=torch.cuda.current_stream(device);side=torch.cuda.Stream(device=device);side.wait_stream(current)
        with torch.cuda.stream(side):
            store_slot(table,index,value);result=load_slot(table,index)
        current.wait_stream(side);torch.cuda.synchronize(device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=side):
            store_slot(table,index,value);result=load_slot(table,index)
        for position in range(50):
            index.fill_(position);graph.replay();torch.cuda.synchronize(device)
            assert torch.equal(result.view(integer),value.view(integer))
            assert torch.equal(table[position].view(integer),value.view(integer));checked+=1
        graph.reset()
    return dict(passed=True,bitwise_graph_copy_slots=checked,dtypes=['float16','float32'],includes_signedzero_infinity_NaN=True)


class Controller:
    def __init__(self,modules,*,steps=50):
        assert steps==50 and modules and len(modules)==len(set(modules))
        self.steps=steps;self.modules=dict(modules);self.saved=[];self.tables={};self.keys={};self.valid={};self.current=None;self.previous=None;self.ready=set();self.counts=Counter();self.completed=[];self.capture_rows=[]
        self.parameters=tuple(p for module in self.modules.values()for p in module.parameters())
        assert self.parameters
        self.master_key=tuple((id(p),p._version,tuple(p.shape),p.dtype,p.device)for p in self.parameters)
        assert len({p.device for p in self.parameters})==1
        self.index=torch.zeros(1,dtype=torch.long,device=self.parameters[0].device)
        for name,module in self.modules.items():
            assert not module.training
            original=module.forward
            def forward(*args,_name=name,_original=original,**kwargs):
                if self.current is None:return _original(*args,**kwargs)
                assert not torch.is_grad_enabled()
                state=self.current;key=(state['stage'],_name)
                if state['read']:
                    assert key in self.tables and state['stage']in self.ready
                    value=load_slot(self.tables[key],self.index)
                else:
                    value=_original(*args,**kwargs)
                    assert isinstance(value,torch.Tensor)and value.is_contiguous()
                    if key not in self.tables:
                        assert not value.is_cuda or not torch.cuda.is_current_stream_capturing(),'Allocate static tables in timed eager warm-up'
                        self.tables[key]=torch.empty((self.steps,*value.shape),dtype=value.dtype,device=value.device)
                    table=self.tables[key]
                    assert table.shape[1:]==value.shape and table.dtype==value.dtype and table.device==value.device
                    store_slot(table,self.index,value)
                if value.is_cuda and torch.cuda.is_current_stream_capturing():
                    self.capture_rows.append(dict(stage=state['stage'],read=state['read'],module=_name,shape=list(value.shape),dtype=str(value.dtype)))
                return value
            module.forward=forward;self.saved.append((module,original))

    def begin(self,*,stage,generation,position,timesteps,fs,backend_key):
        assert self.current is None and stage in('decision','world')and 0<=position<self.steps
        assert not torch.is_grad_enabled()and all(not m.training for m in self.modules.values())
        current_params=tuple(p for module in self.modules.values()for p in module.parameters())
        assert tuple((id(p),p._version,tuple(p.shape),p.dtype,p.device)for p in current_params)==self.master_key,'Static master mutation/replacement requires a new run/cache'
        old=self.previous
        if old is None:assert position==0
        elif old[:2]==(stage,generation):assert position==old[2]+1,'Skipped or repeated DDIM position'
        else:assert old[2]==self.steps-1 and position==0,'Each generation must finish every original step'
        key=(tensor_key(timesteps),tensor_key(fs)if fs is not None else None,backend_key)
        slot=(stage,position)
        if slot in self.keys:assert self.keys[slot]==key,'Time/fs/backend change requires a new run/cache'
        else:self.keys[slot]=key
        read=stage in self.ready
        if read:assert self.valid[stage]==set(range(self.steps))
        self.index.fill_(position)
        self.current=dict(stage=stage,generation=generation,position=position,read=read)
        return read

    def end(self):
        assert self.current is not None
        state=self.current;stage=state['stage'];position=state['position']
        if not state['read']:
            assert all((stage,name)in self.tables for name in self.modules)
            self.valid.setdefault(stage,set()).add(position)
        self.counts[stage+'_calls']+=1;self.counts[stage+('_read'if state['read']else'_fill')]+=1
        self.previous=(stage,state['generation'],position)
        if position==self.steps-1:
            assert self.valid[stage]==set(range(self.steps));self.ready.add(stage)
            self.completed.append(dict(stage=stage,generation=state['generation'],steps=self.steps,read=state['read']))
        self.current=None

    def report(self):
        assert self.current is None
        return dict(exact_static_timestep_cache=True,modules=list(self.modules),counts=dict(self.counts),completed_generations=self.completed,persistent_bytes=sum(t.numel()*t.element_size()for t in self.tables.values()),capture_rows=self.capture_rows.copy(),sampler_steps_reduced=False,rounds_reduced=False,dynamic_features_cached=False,master_schedule_fs_backend_changes_rejected=True,precision_changed=False)

    def close(self):
        assert self.current is None
        for module,forward in self.saved:module.forward=forward
        self.saved.clear()
