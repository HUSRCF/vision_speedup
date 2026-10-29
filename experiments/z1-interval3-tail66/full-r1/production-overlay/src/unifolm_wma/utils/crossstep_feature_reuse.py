"""Experimental two-refresh feature prediction; preserve every DDIM update.

Prediction uses ordinary owned tensors. There is no native graph argument
inspection and no reference/final-video cache. Full closed-loop quality is a
separate requirement from affine toy and graph/kernel fidelity checks.
"""
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import torch
import triton
import triton.language as tl


@contextmanager
def kernel_cache_scope():
    import triton.runtime.cache as cache
    rank=int(os.environ['TEAM07_PARALLEL_RANK']);assert rank in(0,1)
    root=Path(os.environ['TEAM07_PREDICT_CACHE'])/('rank'+str(rank))
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


@triton.jit
def _predict_linear(Current,Previous,Alpha,Live,Output,N:tl.constexpr,
                    RESIDUAL:tl.constexpr,BLOCK:tl.constexpr,EXPERIMENT_TAG:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    current=tl.load(Current+i,i<N,other=0).to(tl.float32)
    previous=tl.load(Previous+i,i<N,other=0).to(tl.float32)
    alpha=tl.load(Alpha)
    value=current+(current-previous)*alpha
    if RESIDUAL:value=value+tl.load(Live+i,i<N,other=0).to(tl.float32)
    tl.store(Output+i,value,i<N)


def predict(current,previous,alpha,live=None,*,output_dtype=None):
    assert current.is_contiguous()and previous.is_contiguous()
    assert current.shape==previous.shape and current.dtype==previous.dtype and current.device==previous.device
    if live is not None:
        assert live.shape==current.shape and live.device==current.device
        live=live.contiguous()
    dtype=output_dtype or current.dtype
    if not current.is_cuda:
        value=current.float()+(current.float()-previous.float())*alpha
        if live is not None:value=value+live.float()
        return value.to(dtype)
    result=torch.empty(current.shape,device=current.device,dtype=dtype)
    with kernel_cache_scope():
        _predict_linear[(triton.cdiv(current.numel(),256),)](
            current,previous,alpha,current if live is None else live,result,current.numel(),
            RESIDUAL=live is not None,BLOCK=256,EXPERIMENT_TAG=1793264401,
            num_warps=4,enable_fp_fusion=False)
    return result


@torch.no_grad()
def graph_check(device):
    cases=[]
    for dtype in(torch.float16,torch.float32):
        for residual in(False,True):
            current=torch.arange(257,device=device,dtype=torch.float32)/256
            previous=current-.125
            current=current.to(torch.float32 if residual else dtype)
            previous=previous.to(current.dtype)
            alpha=torch.tensor([1.],device=device)
            live=(current*.25).to(dtype)if residual else None
            stream=torch.cuda.Stream(device=device);parent=torch.cuda.current_stream(device);stream.wait_stream(parent)
            with torch.cuda.stream(stream):predict(current,previous,alpha,live,output_dtype=dtype)
            parent.wait_stream(stream);torch.cuda.synchronize(device)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):output=predict(current,previous,alpha,live,output_dtype=dtype)
            for scalar in(1.,.5,1/3,2/3,2.):
                current.add_(.25);previous.mul_(.5);alpha.fill_(scalar)
                graph.replay();torch.cuda.synchronize(device)
                expected=current.float()+(current.float()-previous.float())*alpha
                if live is not None:expected=expected+live.float()
                torch.testing.assert_close(output,expected.to(dtype),rtol=0,atol=0)
            graph.reset();cases.append(dict(dtype=str(dtype),residual=residual,tail=257,mutated_replay_bitwise=True,alphas=[1.,.5,1/3,2/3,2.]))
    return dict(passed=True,cases=cases,no_full_model_quality_inferred=True)


class Controller:
    def __init__(self,model,*,input_start=6,output_end=6,interval=2,start=10,stop=40,
                 stages=('world',),boundary_residual=False,scope='deep'):
        assert 0<input_start<len(model.input_blocks)and 0<output_end<len(model.output_blocks)
        assert interval>=2 and 0<start<stop<49 and set(stages)<={'decision','world'}
        assert scope in('deep','spatial')and not(boundary_residual and scope=='spatial')
        self.model=model;self.interval=interval;self.start=start;self.stop=stop;self.stages=tuple(stages)
        self.scope=scope;self.boundary_residual=boundary_residual;self.anchor=None
        self.saved=[];self.buffers={};self.past={};self.valid={};self.current=None;self.previous=None;self.condition=None
        self.counts=Counter();self.generations=[];self.invalidations=Counter();self.eager_counts=Counter()
        self.parameters=tuple(model.parameters());self.identities=tuple(id(p)for p in self.parameters)
        self.versions=None;self.last_invalidation=False;self.shape=None;self.refresh_positions={}
        self.alpha=torch.zeros(1,dtype=torch.float32,device=self.parameters[0].device)
        self.alpha_histogram=Counter();self.copy_check=graph_check(self.alpha.device)if self.alpha.is_cuda else None
        self.entry_name=f'input_blocks.{input_start}';self.exit_name=f'output_blocks.{output_end-1}'
        selected=[(f'input_blocks.{i}',model.input_blocks[i])for i in range(input_start,len(model.input_blocks))]
        selected += [('middle_block',model.middle_block)]
        selected += [(f'output_blocks.{i}',model.output_blocks[i])for i in range(output_end)]
        if scope=='spatial':
            selected=[(n,m)for n,m in model.named_modules()if type(m).__name__=='SpatialTransformer'];assert selected
        for name,module in selected:
            original=module.forward
            def forward(*args,_original=original,_name=name,**kwargs):
                state=self.current
                if state is None or state['stage']not in self.stages:return _original(*args,**kwargs)
                assert not torch.is_grad_enabled()and not self.model.training
                key=(state['stage'],_name);boundary_key=(state['stage'],'boundary_residual')
                if self.boundary_residual and _name==self.entry_name:self.anchor=args[0]
                if state['reuse']:
                    assert self.valid.get(state['stage'],False)
                    self.eager_counts['reuse']+=1
                    if self.scope=='spatial':
                        return self.prediction(key,args[0],output_dtype=args[0].dtype)
                    if self.boundary_residual and _name==self.exit_name:
                        assert self.anchor is not None
                        return self.prediction(boundary_key,self.anchor,output_dtype=self.anchor.dtype)
                    return self.prediction(key)
                value=_original(*args,**kwargs);assert isinstance(value,torch.Tensor)
                if self.scope=='spatial':
                    assert value.shape==args[0].shape and value.dtype==args[0].dtype
                    self.store(key,value.float()-args[0].float())
                else:self.store(key,value)
                if self.boundary_residual and _name==self.exit_name:
                    assert self.anchor is not None and value.shape==self.anchor.shape and value.dtype==self.anchor.dtype
                    self.store(boundary_key,value.float()-self.anchor.float())
                self.eager_counts['refresh']+=1
                return value
            module.forward=forward;self.saved.append((module,original,name))

    def store(self,key,value):
        if key not in self.buffers:
            assert not value.is_cuda or not torch.cuda.is_current_stream_capturing()
            self.buffers[key]=torch.empty_like(value,memory_format=torch.contiguous_format)
            self.past[key]=torch.empty_like(value,memory_format=torch.contiguous_format)
            self.past[key].copy_(value)
        else:self.past[key].copy_(self.buffers[key])
        target=self.buffers[key]
        assert target.shape==value.shape and target.dtype==value.dtype and target.device==value.device
        target.copy_(value)

    def prediction(self,key,live=None,*,output_dtype=None):
        assert key in self.buffers and key in self.past
        return predict(self.buffers[key],self.past[key],self.alpha,live,output_dtype=output_dtype)

    def begin(self,*,stage,generation,position,total_steps,condition_key,input_key):
        assert self.current is None and stage in('decision','world')and total_steps==50 and 0<=position<50
        assert tuple(id(p)for p in self.model.parameters())==self.identities
        assert not self.model.training and not torch.is_grad_enabled()
        old=self.previous
        if old is None:assert position==0
        elif old[:2]==(stage,generation):assert position==old[2]+1
        else:assert old[2]==49 and position==0
        versions=tuple(p._version for p in self.parameters);reason=None
        if self.shape is None:self.shape=input_key
        else:assert self.shape==input_key
        if position==0:reason='new_generation'
        elif versions!=self.versions:reason='master_version'
        elif condition_key!=self.condition:reason='condition_change'
        self.last_invalidation=reason in('master_version','condition_change')
        if reason is not None:
            self.valid[stage]=False;self.refresh_positions[stage]=[];self.invalidations[reason]+=1
        history=self.refresh_positions.get(stage,[])
        reuse=stage in self.stages and self.start<=position<self.stop and(position-self.start)%self.interval!=0 and self.valid.get(stage,False)and len(history)==2
        alpha=0.
        if reuse:
            before,last=history;assert before<last<position
            alpha=(position-last)/(last-before);self.alpha_histogram[str(alpha)]+=1
        self.alpha.fill_(alpha)
        self.current=dict(stage=stage,generation=generation,position=position,reuse=reuse)
        self.anchor=None;self.versions=versions;self.condition=condition_key
        return reuse

    def end(self):
        assert self.current is not None
        state=self.current;stage=state['stage'];position=state['position'];self.valid[stage]=True
        if stage in self.stages and not state['reuse']:
            self.refresh_positions[stage]=(self.refresh_positions.get(stage,[])+[position])[-2:]
        self.counts[stage+'_calls']+=1;self.counts[stage+('_reuse'if state['reuse']else'_refresh')]+=1
        self.previous=(stage,state['generation'],position);self.current=None;self.anchor=None
        if position==49:self.generations.append(dict(stage=stage,generation=state['generation'],steps=50))

    def report(self):
        assert self.current is None
        return dict(experimental_dynamic_feature_reuse=True,predictive_feature_reuse=True,history_complete_refreshes=2,
            prediction='current + (current - previous) * actual_position_gap_ratio',alpha_histogram=dict(self.alpha_histogram),
            residual_cache_FP32=True,graph_prediction_selfcheck=self.copy_check,interval=self.interval,window=[self.start,self.stop],
            stages=list(self.stages),selected_blocks=[n for _,_,n in self.saved],counts=dict(self.counts),completed_generations=self.generations,
            invalidations=dict(self.invalidations),persistent_feature_bytes=sum(t.numel()*t.element_size()for d in(self.buffers,self.past)for t in d.values()),
            eager_or_capture_block_calls=dict(self.eager_counts),sampler_step_count_reduced=False,round_count_reduced=False,
            shallow_blocks_and_heads_live=self.scope=='deep',no_reference_or_final_output_cache=True,full_video_PSNR_required=True,
            boundary_residual=self.boundary_residual,current_boundary_input_live=self.boundary_residual,scope=self.scope,
            all_convolutions_and_heads_live=self.scope=='spatial',per_module_residual=self.scope=='spatial')

    def close(self):
        assert self.current is None
        for module,original,_ in self.saved:module.forward=original
        self.saved.clear()
