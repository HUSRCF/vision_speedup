"""Exact per-stage, per-timestep memoization of explicitly static MLP outputs.

Only register the WMA time/fps embeddings and ResBlock.emb_layers. Dynamic
video/action/state features, attention and sampler updates are never cached.
All allocation/initial filling remains in the original process timer.
"""
from collections import Counter
import torch


def tensor_key(tensor):
    assert isinstance(tensor,torch.Tensor)and tensor.numel()>0
    # Read only tiny declared sampler timesteps/fs, never a feature or weight.
    return tuple(tensor.shape),str(tensor.dtype),tuple(tensor.detach().cpu().reshape(-1).tolist())


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
                    value=self.tables[key].index_select(0,self.index).squeeze(0)
                else:
                    value=_original(*args,**kwargs)
                    assert isinstance(value,torch.Tensor)and value.is_contiguous()
                    if key not in self.tables:
                        assert not value.is_cuda or not torch.cuda.is_current_stream_capturing(),'Allocate static tables in timed eager warm-up'
                        self.tables[key]=torch.empty((self.steps,*value.shape),dtype=value.dtype,device=value.device)
                    table=self.tables[key]
                    assert table.shape[1:]==value.shape and table.dtype==value.dtype and table.device==value.device
                    table.index_copy_(0,self.index,value.unsqueeze(0))
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
