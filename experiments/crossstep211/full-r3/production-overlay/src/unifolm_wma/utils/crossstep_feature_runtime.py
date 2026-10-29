"""Keep all50 sampler updates; choose a refresh or feature-reuse CUDA graph."""
import json,os
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten
from .crossstep_feature_reuse import Controller

def condition_key(args,kwargs):
    leaves,spec=tree_flatten((args[4],kwargs.get('fs')))
    return str(spec),tuple((id(v),v._version,tuple(v.shape),v.dtype,v.device)if isinstance(v,torch.Tensor)else(type(v).__name__,v)for v in leaves)

class Runtime:
    def __init__(self,model,*,input_start=6,output_end=6,start=10,stop=40):
        from unifolm_wma.utils.cuda_graph import DenoiserGraph
        self.model=model;self.graph=model.apply_model;assert isinstance(self.graph,DenoiserGraph) and not self.graph.entries
        self.graph.max_graphs=3
        self.controller=Controller(model.model.diffusion_model,input_start=input_start,output_end=output_end,start=start,stop=stop)
        self.step=None;self.generation={'decision':-1,'world':-1};self.step_calls=0;self.refresh_pool=None
        model.apply_model=self.apply

    def set_step(self,stage,position):
        assert self.step is None and 0<=position<50
        if position==0:self.generation[stage]+=1
        self.step=(stage,self.generation[stage],position);self.step_calls=0

    def finish_step(self):
        assert self.step is not None and self.step_calls==1,'Expected one denoiser call per unchanged DDIM step'
        self.step=None

    def apply(self,*args,**kwargs):
        assert self.step is not None and self.step_calls==0
        stage,generation,position=self.step
        reuse=self.controller.begin(stage=stage,generation=generation,position=position,total_steps=50,condition_key=condition_key(args,kwargs),input_key=(tuple(args[0].shape),args[0].dtype,args[0].device))
        if self.controller.last_invalidation:
            for entry in self.graph.entries.values():entry['condition_identity']=None
        assert '_crossstep_feature_reuse' not in kwargs
        kwargs=dict(kwargs,_crossstep_feature_reuse=reuse)
        before=set(self.graph.entries)
        # Feature buffers are allocated outside the graph pools. Full and reuse
        # graphs execute sequentially, with outputs cloned by DenoiserGraph.
        self.graph.experimental_capture_pool=self.refresh_pool if reuse else None
        # The optional head-filter cache has a full-graph-only 23-convolution
        # contract. A reuse graph is intentionally smaller: retain its original
        # native head transforms instead of relaxing that cache's assertions.
        flag='UNIFOLM_WORLD_HEAD_FILTER_CACHE';previous_flag=os.environ.get(flag)
        if reuse:os.environ[flag]='0'
        try:
            result=self.graph(*args,**kwargs)
            for key in set(self.graph.entries)-before:
                entry=self.graph.entries[key];entry['crossstep_role']=(stage,reuse)
                if stage=='world' and not reuse:self.refresh_pool=entry['graph'].pool()
        finally:
            self.graph.experimental_capture_pool=None
            if previous_flag is None:os.environ.pop(flag,None)
            else:os.environ[flag]=previous_flag
        self.controller.end();self.step_calls+=1
        return result

    def report(self):
        assert self.step is None
        return dict(self.controller.report(),graph_count=len(self.graph.entries),graph_roles=[list(e['crossstep_role'])for e in self.graph.entries.values()],world_refresh_and_reuse_share_graph_pool=True,reuse_graph_head_filter_cache=False,reuse_graph_native_head_transforms_live=True,full_refresh_graph_head_cache_unchanged=True)

active=None
parent_sampler=None

def install(model,**options):
    global active,parent_sampler
    from unifolm_wma.models.samplers.ddim import DDIMSampler
    assert active is None;active=Runtime(model,**options);parent_sampler=DDIMSampler.p_sample_ddim
    def sample(sampler,*args,**kwargs):
        if sampler.model is not model:return parent_sampler(sampler,*args,**kwargs)
        assert len(sampler.ddim_timesteps)==50 and 'index' in kwargs
        stage='decision'if torch.backends.cudnn.allow_tf32 else'world'
        position=49-int(kwargs['index']);active.set_step(stage,position)
        result=parent_sampler(sampler,*args,**kwargs);active.finish_step();return result
    DDIMSampler.p_sample_ddim=sample
    print('CROSSSTEP_FEATURE_REUSE_INSTALLED '+json.dumps(dict(input_start=options.get('input_start',6),output_end=options.get('output_end',6),window=[options.get('start',10),options.get('stop',40)],interval=2,stages=['world'],sampler_steps=50,required_rounds_preserved=True,graph_limit=3)),flush=True)
    return active

def finish(args):
    assert active is not None
    report=active.report();assert report['counts']['decision_calls']==report['counts']['world_calls']==50*args.n_iter
    assert len(report['completed_generations'])==2*args.n_iter and report['counts']['world_reuse']>0
    path=Path(args.profile_path).with_suffix('.crossstep.json');path.write_text(json.dumps(report,indent=2)+'\n')
    print('CROSSSTEP_FEATURE_REUSE_COMPLETE '+json.dumps(report),flush=True)
