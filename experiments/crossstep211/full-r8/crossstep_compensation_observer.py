"""Read-only temporal correlation of normal Torch FFN high/low tensors.

No native graph arguments are inspected. No input, weight, feature array or
address is exported. Full original inference remains live and is quality-checked.
"""
from collections import Counter
import json
from pathlib import Path
import torch

def cosine(a,b):
    return float((a.double()*b.double()).sum()/(a.double().norm()*b.double().norm()).clamp_min(1e-30))

def sample(high,low,wh,wl,previous):
    h=high[:32].float();l=low[:32].float();w=wh[:64].float();wlow=wl[:64].float()
    primary=(h@w.T)*(1./256.)
    correction=((l@w.T)+(h@wlow.T))*(1./(2048.*256.))
    scale=(primary+correction).double().norm().clamp_min(1e-30)
    assert torch.isfinite(primary).all()and torch.isfinite(correction).all()
    result=dict(correction_relative_l2=float(correction.double().norm()/scale))
    if previous is not None:
        ph,pl,pc=previous
        error=(correction-pc).double().norm()
        result.update(high_cosine=cosine(h,ph),low_cosine=cosine(l,pl),correction_cosine=cosine(correction,pc),reuse_relative_l2=float(error/scale),reuse_vs_drop_error_ratio=float(error/correction.double().norm().clamp_min(1e-30)))
    return result,(h.clone(),l.clone(),correction.clone())

class Observer:
    def __init__(self,model):
        from unifolm_wma.utils.cuda_graph import DenoiserGraph
        from unifolm_wma.utils import world_geglu_hidden_parts_c1280_zero as owner_module
        self.model=model;self.graph=model.apply_model;assert isinstance(self.graph,DenoiserGraph)and not self.graph.entries
        self.owner_module=owner_module;self.original=owner_module.zero_ffn1280_hidden_residual
        self.linears=[(n,m)for n,m in model.model.diffusion_model.named_modules()if isinstance(m,torch.nn.Linear)and tuple(m.weight.shape)==(1280,5120)]
        self.weight_parts=None;self.master=None;self.master_version=None;self.name=None;self.inputs=None
        self.step=None;self.calls=0;self.generation={'decision':-1,'world':-1};self.previous_step=None;self.previous_sample=None
        self.counts=Counter();self.completed=[];self.rows=[];self.weight_low_nonzero=None
        owner_module.zero_ffn1280_hidden_residual=self.observe;model.apply_model=self.apply

    def observe(self,inputs,parts,count,residual,bias=None):
        if self.step is not None and self.step[0]=='world'and self.step[1]==0 and tuple(inputs[0].shape)==(2560,5120):
            if self.weight_parts is None:
                assert not torch.cuda.is_current_stream_capturing()
                matches=[]
                for name,module in self.linears:
                    cache=getattr(module,'_decision_ffn_residual_weight_cache',None)
                    if cache is not None and cache[1][0]is parts[0]and cache[1][1]is parts[1]:matches.append((name,module))
                assert len(matches)==1
                self.name,module=matches[0];self.master=module.weight;self.master_version=self.master._version;self.weight_parts=parts
            if self.weight_parts[0]is parts[0]and torch.cuda.is_current_stream_capturing():
                assert self.inputs is None,'One capture of the selected normal Torch owner expected'
                self.inputs=tuple(inputs)
        return self.original(inputs,parts,count,residual,bias)

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
        result=self.graph(*args,**kwargs);stage,generation,position=self.step
        if stage=='world'and generation==0:
            assert self.inputs is not None and self.master._version==self.master_version
            before=torch.cuda.get_rng_state().clone()
            row,self.previous_sample=sample(*self.inputs,*self.weight_parts,self.previous_sample)
            assert torch.equal(before,torch.cuda.get_rng_state())
            row['position']=position;self.rows.append(row)
            if position==0:self.weight_low_nonzero=int(torch.count_nonzero(self.weight_parts[1]))
        self.calls+=1;self.counts[stage]+=1;return result

    def end_step(self):
        assert self.step is not None and self.calls==1
        stage,generation,position=self.step
        if position==49:self.completed.append(dict(stage=stage,generation=generation,steps=50))
        self.previous_step=self.step;self.step=None

    def report(self):
        assert self.step is None and len(self.rows)==50
        return dict(diagnostic_only=True,read_only_compensation_correlation=True,normal_Torch_function_hook_only=True,no_native_argument_inspection=True,no_tensors_weights_or_addresses_exported=True,inference_outputs_unmodified=True,counts=dict(self.counts),completed_generations=self.completed,graph_count=len(self.graph.entries),owner=self.name,full_shape=[2560,1280,5120],sample_shape=[32,64,5120],weight_low_nonzero=self.weight_low_nonzero,rows=self.rows,RNG_unchanged=True,sampler_step_count_reduced=False,round_count_reduced=False,diagnostic_matmuls_are_FP32_not_exact_original_BK64_rounding=True,full_video_PSNR_required=True,official_adopted=False)

active=None
parent_sampler=None

def install(model,**options):
    global active,parent_sampler
    from unifolm_wma.models.samplers.ddim import DDIMSampler
    assert active is None;active=Observer(model);parent_sampler=DDIMSampler.p_sample_ddim
    def sample_step(sampler,*args,**kwargs):
        if sampler.model is not model:return parent_sampler(sampler,*args,**kwargs)
        assert len(sampler.ddim_timesteps)==50
        stage='decision'if torch.backends.cudnn.allow_tf32 else'world';active.set_step(stage,49-int(kwargs['index']))
        result=parent_sampler(sampler,*args,**kwargs);active.end_step();return result
    DDIMSampler.p_sample_ddim=sample_step
    print('COMPENSATION_CORRELATION_INSTALLED original_inference=true first_world50_steps_only=true',flush=True)

def finish(args):
    report=active.report();assert report['counts']==dict(decision=50*args.n_iter,world=50*args.n_iter)and len(report['completed_generations'])==2*args.n_iter and report['graph_count']==2
    path=Path(args.profile_path).with_suffix('.crossstep.json');path.write_text(json.dumps(report,indent=2)+'\n')
    active.owner_module.zero_ffn1280_hidden_residual=active.original
    print('COMPENSATION_CORRELATION_COMPLETE '+json.dumps(report),flush=True)
