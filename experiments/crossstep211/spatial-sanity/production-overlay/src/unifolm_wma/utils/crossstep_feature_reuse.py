"""Experimental dynamic U-Net feature reuse, preserving every sampler call.

Only selected deep blocks are cached. Shallow blocks, output heads, sampler
updates and RNG remain live. Full video quality is an independent requirement.
"""
from collections import Counter
import torch

class Controller:
    def __init__(self,model,*,input_start=6,output_end=6,interval=2,start=10,stop=40,stages=('world',),boundary_residual=False,scope='deep'):
        assert 0<input_start<len(model.input_blocks) and 0<output_end<len(model.output_blocks)
        assert interval>=2 and 0<start<stop<49 and set(stages)<={'decision','world'}
        self.model=model;self.interval=interval;self.start=start;self.stop=stop;self.stages=tuple(stages)
        self.saved=[];self.buffers={};self.valid={};self.current=None;self.previous=None;self.condition=None
        self.counts=Counter();self.generations=[];self.invalidations=Counter();self.eager_counts=Counter()
        self.parameters=tuple(model.parameters());self.identities=tuple(id(p)for p in self.parameters)
        self.versions=None;self.last_invalidation=False;self.shape=None
        self.boundary_residual=boundary_residual;self.anchor=None
        assert scope in ('deep','spatial')and not(boundary_residual and scope=='spatial')
        self.scope=scope
        self.entry_name=f'input_blocks.{input_start}';self.exit_name=f'output_blocks.{output_end-1}'
        selected=[(f'input_blocks.{i}',model.input_blocks[i])for i in range(input_start,len(model.input_blocks))]
        selected += [('middle_block',model.middle_block)]
        selected += [(f'output_blocks.{i}',model.output_blocks[i])for i in range(output_end)]
        if scope=='spatial':
            selected=[(name,module)for name,module in model.named_modules()if type(module).__name__=='SpatialTransformer']
            assert selected,'No SpatialTransformer modules'
        for name,module in selected:
            original=module.forward
            def forward(*args,_original=original,_name=name,**kwargs):
                state=self.current
                if state is None or state['stage'] not in self.stages:return _original(*args,**kwargs)
                assert not torch.is_grad_enabled() and not self.model.training
                key=(state['stage'],_name)
                if self.boundary_residual and _name==self.entry_name:self.anchor=args[0]
                boundary_key=(state['stage'],'boundary_residual')
                if state['reuse']:
                    if self.scope=='spatial':
                        assert self.valid.get(state['stage'],False)and key in self.buffers
                        assert args[0].shape==self.buffers[key].shape and args[0].dtype==self.buffers[key].dtype
                        self.eager_counts['reuse']+=1
                        return args[0]+self.buffers[key]
                    if self.boundary_residual and _name==self.exit_name:
                        assert self.valid.get(state['stage'],False) and boundary_key in self.buffers
                        delta=self.buffers[boundary_key];assert self.anchor is not None and delta.shape==self.anchor.shape and delta.dtype==self.anchor.dtype
                        self.eager_counts['reuse']+=1
                        return self.anchor+delta
                    assert self.valid.get(state['stage'],False) and key in self.buffers,'Cache miss must execute a full refresh'
                    self.eager_counts['reuse']+=1
                    return self.buffers[key].clone()
                value=_original(*args,**kwargs);assert isinstance(value,torch.Tensor)
                if key not in self.buffers:
                    assert not value.is_cuda or not torch.cuda.is_current_stream_capturing(),'Allocate feature buffers during timed eager warm-up'
                    self.buffers[key]=torch.empty_like(value)
                target=self.buffers[key]
                assert target.shape==value.shape and target.dtype==value.dtype and target.device==value.device,'Shape/dtype/device change requires graph rebuild'
                if self.scope=='spatial':
                    assert value.shape==args[0].shape and value.dtype==args[0].dtype,'Spatial residual must preserve shape/dtype'
                    target.copy_(value-args[0])
                else:target.copy_(value)
                self.eager_counts['refresh']+=1
                if self.boundary_residual and _name==self.exit_name:
                    assert self.anchor is not None and value.shape==self.anchor.shape and value.dtype==self.anchor.dtype,'Residual boundary must preserve shape/dtype'
                    if boundary_key not in self.buffers:
                        assert not value.is_cuda or not torch.cuda.is_current_stream_capturing()
                        self.buffers[boundary_key]=torch.empty_like(value)
                    self.buffers[boundary_key].copy_(value-self.anchor)
                return value
            module.forward=forward;self.saved.append((module,original,name))

    def begin(self,*,stage,generation,position,total_steps,condition_key,input_key):
        assert self.current is None and stage in ('decision','world')
        assert total_steps==50 and 0<=position<50
        assert tuple(id(p)for p in self.model.parameters())==self.identities,'Parameter replacement requires graph rebuild'
        assert not self.model.training and not torch.is_grad_enabled()
        old=self.previous;scope=(stage,generation)
        if old is None:assert position==0
        elif old[:2]==scope:assert position==old[2]+1,'Skipped or repeated sampler position'
        else:assert old[2]==49 and position==0,'Generation must retain all 50 sampler positions'
        versions=tuple(p._version for p in self.parameters);reason=None
        if self.shape is None:self.shape=input_key
        else:assert self.shape==input_key,'Input shape change requires graph rebuild'
        if position==0:reason='new_generation'
        elif versions!=self.versions:reason='master_version'
        elif condition_key!=self.condition:reason='condition_change'
        self.last_invalidation=reason in ('master_version','condition_change')
        if reason is not None:self.valid[stage]=False;self.invalidations[reason]+=1
        reuse=stage in self.stages and self.start<=position<self.stop and (position-self.start)%self.interval!=0 and self.valid.get(stage,False)
        self.current=dict(stage=stage,generation=generation,position=position,reuse=reuse)
        self.anchor=None
        self.versions=versions;self.condition=condition_key
        return reuse

    def end(self):
        assert self.current is not None
        state=self.current;stage=state['stage'];self.valid[stage]=True
        self.counts[stage+'_calls']+=1;self.counts[stage+('_reuse'if state['reuse']else'_refresh')]+=1
        self.previous=(stage,state['generation'],state['position']);self.current=None
        self.anchor=None
        if state['position']==49:self.generations.append(dict(stage=stage,generation=state['generation'],steps=50))

    def report(self):
        assert self.current is None
        return dict(experimental_dynamic_feature_reuse=True,interval=self.interval,window=[self.start,self.stop],stages=list(self.stages),selected_blocks=[name for _,_,name in self.saved],counts=dict(self.counts),completed_generations=self.generations,invalidations=dict(self.invalidations),persistent_feature_bytes=sum(t.numel()*t.element_size()for t in self.buffers.values()),eager_or_capture_block_calls=dict(self.eager_counts),sampler_step_count_reduced=False,round_count_reduced=False,shallow_blocks_and_heads_live=self.scope=='deep',no_reference_or_final_output_cache=True,full_video_PSNR_required=True,boundary_residual=self.boundary_residual,current_boundary_input_live=self.boundary_residual,scope=self.scope,all_convolutions_and_heads_live=self.scope=='spatial',per_module_residual=self.scope=='spatial')

    def close(self):
        assert self.current is None
        for module,original,_ in self.saved:module.forward=original
        self.saved.clear()
