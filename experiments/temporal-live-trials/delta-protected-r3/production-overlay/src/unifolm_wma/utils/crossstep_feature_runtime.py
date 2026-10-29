"""Experimental live FFN delta updates inside the original two denoiser graphs."""
from collections import Counter
import json,os
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten
from unifolm_wma.utils import temporal_delta_gemm_candidate as candidate


def reuse_position(position):return 10<=position<40 and(position-10)%2==1


def can_delta(name,kind):
    return kind=='SpatialTransformer'and not name.startswith('middle_block.')


def condition_key(args,kwargs):
    leaves,spec=tree_flatten((args[4],kwargs.get('fs')))
    return str(spec),tuple((id(v),v._version,tuple(v.shape),v.dtype,v.device)if isinstance(v,torch.Tensor)else(type(v).__name__,v)for v in leaves)


class Runtime:
    def __init__(self,model):
        from unifolm_wma.utils.cuda_graph import DenoiserGraph
        from unifolm_wma.utils import world_geglu_hidden_parts_c1280_zero as owner_module
        self.graph=model.apply_model;assert isinstance(self.graph,DenoiserGraph)and not self.graph.entries
        self.owner_module=owner_module;self.original=owner_module.zero_ffn1280_hidden_residual
        self.linears=[(n,m)for n,m in model.model.diffusion_model.named_modules()if isinstance(m,torch.nn.Linear)and tuple(m.weight.shape)==(1280,5120)]
        self.owners={};self.ineligible=set();self.protected={};self.diffusion=model.model.diffusion_model;self.counts=Counter();self.completed=[];self.invalidations=Counter()
        self.step=None;self.calls=0;self.previous_step=None;self.generation={'decision':-1,'world':-1};self.condition=None;self.valid=False
        self.flag=torch.zeros((),device=next(model.parameters()).device,dtype=torch.int32)
        self.stages=int(os.environ.get('UNIFOLM_WORLD_FFN1280_ZERO_STAGES','2'));assert self.stages in(2,3)
        owner_module.zero_ffn1280_hidden_residual=self.forward;model.apply_model=self.apply

    def forward(self,inputs,parts,count,residual,bias=None):
        if self.step is None or self.step[0]!='world' or inputs[0].shape[0]not in(640,2560):return self.original(inputs,parts,count,residual,bias)
        key=id(parts[0]);capturing=torch.cuda.is_current_stream_capturing()
        if key in self.ineligible:return self.original(inputs,parts,count,residual,bias)
        if key not in self.owners:
            assert not capturing,'Owner and delta buffers must be prepared during timed eager warmup'
            if int(count)!=0:self.ineligible.add(key);return self.original(inputs,parts,count,residual,bias)
            matches=[]
            for name,module in self.linears:
                cache=getattr(module,'_decision_ffn_residual_weight_cache',None)
                if cache is not None and cache[1][0]is parts[0]and cache[1][1]is parts[1]:matches.append((name,module))
            assert len(matches)==1
            name,module=matches[0]
            prefix=name.split('.transformer_blocks.',1)[0]
            kind=type(self.diffusion.get_submodule(prefix)).__name__
            if not can_delta(name,kind):
                self.ineligible.add(key);self.protected[name]=kind
                return self.original(inputs,parts,count,residual,bias)
            master=module.weight
            self.owners[key]=dict(name=name,module=module,master=master,version=master._version,shape=tuple(inputs[0].shape),parts=parts,cache=candidate.allocate(inputs[0],1280),captures=0)
        owner=self.owners[key]
        assert owner['shape']==tuple(inputs[0].shape)and owner['parts'][0]is parts[0]and owner['parts'][1]is parts[1]
        if capturing:owner['captures']+=1
        output=torch.empty_like(residual)
        bm=64 if inputs[0].shape[0]>=2560 else 32
        return candidate.run_device(inputs,parts,count,bias,residual,output,owner['cache'],self.flag,bm=bm,stages=self.stages)

    def set_step(self,stage,position):
        assert self.step is None
        if position==0:self.generation[stage]+=1
        current=(stage,self.generation[stage],position)
        if self.previous_step is None:assert position==0
        elif self.previous_step[:2]==current[:2]:assert position==self.previous_step[2]+1
        else:assert self.previous_step[2]==49 and position==0
        self.step=current;self.calls=0

    def apply(self,*args,**kwargs):
        assert self.step is not None and self.calls==0
        stage,generation,position=self.step
        for owner in self.owners.values():
            assert owner['module'].weight is owner['master']and owner['master']._version==owner['version'],'Changed masters require graph/cache rebuild'
        signature=condition_key(args,kwargs)
        if stage=='world':
            if position==0:self.valid=False;self.invalidations['new_world_generation']+=1
            elif signature!=self.condition:self.valid=False;self.invalidations['condition_change']+=1
        reuse=stage=='world'and reuse_position(position)and self.valid
        self.flag.fill_(int(reuse))
        result=self.graph(*args,**kwargs)
        if stage=='world':self.valid=True;self.condition=signature
        self.counts[stage+'_calls']+=1;self.counts[stage+('_reuse'if reuse else'_refresh')]+=1
        self.calls+=1;return result

    def end_step(self):
        assert self.step is not None and self.calls==1
        stage,generation,position=self.step
        if position==49:self.completed.append(dict(stage=stage,generation=generation,steps=50))
        self.previous_step=self.step;self.step=None

    def report(self):
        assert self.step is None and self.owners and all(o['captures']==1 for o in self.owners.values())
        owners=[dict(name=o['name'],shape=[o['shape'][0],1280,5120],captures=o['captures'])for o in self.owners.values()]
        return dict(temporal_delta_runtime=True,protected_spatial_only=True,protected_owners=[dict(name=n,transformer=k)for n,k in sorted(self.protected.items())],counts=dict(self.counts),completed_generations=self.completed,graph_count=len(self.graph.entries),owners=owners,owner_count=len(owners),window=[10,40],interval=2,stages=['world'],persistent_cache_bytes=sum(t.numel()*t.element_size()for o in self.owners.values()for t in o['cache'].values()),invalidations=dict(self.invalidations),fresh_current_inputs_bias_and_residual=True,cached_prebias_product_only=True,weight_low_nonzero_falls_back=True,master_changes_require_graph_rebuild=True,condition_and_generation_invalidation=True,sampler_step_count_reduced=False,round_count_reduced=False,no_native_argument_inspection=True,no_weights_or_features_exported=True,full_video_PSNR_required=True,official_adopted=False)


active=None
parent_sampler=None


def install(model):
    global active,parent_sampler
    from unifolm_wma.models.samplers.ddim import DDIMSampler
    assert active is None;active=Runtime(model);parent_sampler=DDIMSampler.p_sample_ddim
    def sample_step(sampler,*args,**kwargs):
        if sampler.model is not model:return parent_sampler(sampler,*args,**kwargs)
        assert len(sampler.ddim_timesteps)==50
        condition=args[3]if len(args)>3 else kwargs['c'];sim_mode=condition['c_crossattn_action'][2];assert isinstance(sim_mode,bool)
        active.set_step('world'if sim_mode else'decision',49-int(kwargs['index']))
        result=parent_sampler(sampler,*args,**kwargs);active.end_step();return result
    DDIMSampler.p_sample_ddim=sample_step
    print('TEMPORAL_DELTA_RUNTIME_INSTALLED eligible_spatial_C1280_consumers_only=true original_temporal_and_middle=true two_original_graphs=true',flush=True)


def finish(args):
    report=active.report();assert report['counts']['decision_calls']==report['counts']['world_calls']==50*args.n_iter and report['counts']['world_reuse']==15*args.n_iter and report['counts'].get('decision_reuse',0)==0
    assert len(report['completed_generations'])==2*args.n_iter and report['graph_count']==2
    Path(args.profile_path).with_suffix('.crossstep.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TEMPORAL_DELTA_RUNTIME_COMPLETE '+json.dumps(report),flush=True)
