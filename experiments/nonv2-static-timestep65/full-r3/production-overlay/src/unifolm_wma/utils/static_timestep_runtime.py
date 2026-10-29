"""Keep every DDIM call; memoize only source-verified static time MLPs."""
import json
from pathlib import Path
import torch
from .static_timestep_cache import Controller,graph_copy_selfcheck


def select_modules(backbone):
    from unifolm_wma.modules.networks.wma_model import ResBlock
    modules={'time_embed':backbone.time_embed}
    if backbone.fs_condition:modules['fps_embedding']=backbone.fps_embedding
    for name,module in backbone.named_modules():
        if type(module)is ResBlock:modules[name+'.emb_layers']=module.emb_layers
    assert len(modules)>2
    for name,module in modules.items():
        assert type(module)is torch.nn.Sequential and not module.training
        children=tuple(type(m)for m in module.children())
        assert children==((torch.nn.SiLU,torch.nn.Linear)if name.endswith('.emb_layers')else(torch.nn.Linear,torch.nn.SiLU,torch.nn.Linear))
    return modules


class Runtime:
    def __init__(self,model):
        from unifolm_wma.utils.cuda_graph import DenoiserGraph
        self.model=model;self.executor=model.apply_model;self.graph=getattr(self.executor,'denoiser_graph',self.executor)
        assert isinstance(self.graph,DenoiserGraph)and not self.graph.entries
        self.copy_check=graph_copy_selfcheck(next(model.model.diffusion_model.parameters()).device)if next(model.model.diffusion_model.parameters()).is_cuda else None
        self.graph.max_graphs=4;self.controller=Controller(select_modules(model.model.diffusion_model));self.step=None;self.calls=0;self.generations={'decision':-1,'world':-1};self.pools={}
        self.default_fs=model.model.diffusion_model.default_fs;self.fs_condition=model.model.diffusion_model.fs_condition
        self.input_key=None;model.apply_model=self.apply

    def set_step(self,stage,position):
        assert self.step is None and stage in self.generations and 0<=position<50
        if position==0:self.generations[stage]+=1
        self.step=(stage,self.generations[stage],position);self.calls=0

    def finish_step(self):
        assert self.step is not None and self.calls==1
        self.step=None

    def apply(self,*args,**kwargs):
        assert self.step is not None and self.calls==0 and len(args)>=5
        backbone=self.model.model.diffusion_model
        assert backbone.default_fs==self.default_fs and backbone.fs_condition==self.fs_condition
        stage,generation,position=self.step
        shape=(tuple(args[0].shape),args[0].dtype,args[0].device)
        if self.input_key is None:self.input_key=shape
        else:assert shape==self.input_key,'Input geometry change requires a new cache/run'
        read=self.controller.begin(stage=stage,generation=generation,position=position,timesteps=args[3],fs=kwargs.get('fs'),backend_key=self.graph.backend_math_key())
        assert '_static_timestep_read'not in kwargs;kwargs=dict(kwargs,_static_timestep_read=read)
        before=set(self.graph.entries);self.graph.experimental_capture_pool=self.pools.get(stage)if read else None
        try:
            result=self.executor(*args,**kwargs)
            for key in set(self.graph.entries)-before:
                entry=self.graph.entries[key];entry['static_time_role']=(stage,read)
                if not read:self.pools[stage]=entry['graph'].pool()
        finally:self.graph.experimental_capture_pool=None
        self.controller.end();self.calls+=1;return result

    def report(self):
        assert self.step is None
        return dict(self.controller.report(),stages=['decision','world'],graph_count=len(self.graph.entries),graph_roles=[list(e['static_time_role'])for e in self.graph.entries.values()],parallel_executor_preserved=True,full_and_read_share_per_stage_pool=True,graph_copy_selfcheck=self.copy_check)

active=None
parent_sampler=None


def install(model):
    global active,parent_sampler
    from unifolm_wma.models.samplers.ddim import DDIMSampler
    assert active is None;active=Runtime(model);parent_sampler=DDIMSampler.p_sample_ddim
    def sample(sampler,*args,**kwargs):
        if sampler.model is not model:return parent_sampler(sampler,*args,**kwargs)
        assert len(sampler.ddim_timesteps)==50 and 'index'in kwargs
        condition=args[3]if len(args)>3 else kwargs['c'];sim_mode=condition['c_crossattn_action'][2];assert isinstance(sim_mode,bool)
        stage='world'if sim_mode else'decision';active.set_step(stage,49-int(kwargs['index']))
        result=parent_sampler(sampler,*args,**kwargs);active.finish_step();return result
    DDIMSampler.p_sample_ddim=sample
    print('STATIC_TIMESTEP_CACHE_INSTALLED '+json.dumps(dict(modules=list(active.controller.modules),sampler_steps=50,precision_changed=False,all_rounds_preserved=True)),flush=True)
    return active


def finish(args):
    assert active is not None
    report=active.report();assert report['graph_count']==4
    counts=report['counts'];assert counts['decision_calls']==counts['world_calls']==50*args.n_iter
    assert counts['decision_fill']==counts['world_fill']==50
    assert counts['decision_read']==counts['world_read']==50*(args.n_iter-1)
    assert len(report['completed_generations'])==2*args.n_iter
    rows=report['capture_rows'];expected={(stage,read,name)for stage in('decision','world')for read in(False,True)for name in report['modules']}
    assert len(rows)==len(expected)and {(r['stage'],r['read'],r['module'])for r in rows}==expected
    path=Path(args.profile_path).with_suffix('.static-time.json');path.write_text(json.dumps(report,indent=2)+'\n')
    print('STATIC_TIMESTEP_CACHE_COMPLETE '+json.dumps(report),flush=True)
