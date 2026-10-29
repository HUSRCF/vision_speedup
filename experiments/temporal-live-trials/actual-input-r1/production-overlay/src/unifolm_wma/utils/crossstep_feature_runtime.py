"""Measure actual consecutive FFN operands without changing inference outputs.

Only normal Torch function arguments and registered module caches are used.
All operands stay on the GPU; exported evidence is scalar errors and timings.
"""
from collections import Counter
import json,os,statistics
from pathlib import Path
import torch
import triton
from unifolm_wma.utils import temporal_delta_gemm_candidate as candidate
from unifolm_wma.utils.ffn1280_zero_residual import _zero_low_gemm_residual


def errors(reference,actual):
    difference=(actual.double()-reference.double())
    return dict(relative_l2=float(difference.norm()/reference.double().norm().clamp_min(1e-30)),max_abs=float(difference.abs().max()),finite=bool(torch.isfinite(actual).all()))


class Observer:
    def __init__(self,model):
        from unifolm_wma.utils.cuda_graph import DenoiserGraph
        from unifolm_wma.utils import world_geglu_hidden_parts_c1280_zero as owner_module
        self.graph=model.apply_model;assert isinstance(self.graph,DenoiserGraph)and not self.graph.entries
        self.owner_module=owner_module;self.original=owner_module.zero_ffn1280_hidden_residual
        self.linears=[(n,m)for n,m in model.model.diffusion_model.named_modules()if isinstance(m,torch.nn.Linear)and tuple(m.weight.shape)==(1280,5120)]
        self.parts=None;self.inputs=None;self.count=None;self.name=None;self.master=None;self.master_version=None
        self.step=None;self.calls=0;self.generation={'decision':-1,'world':-1};self.previous_step=None
        self.counts=Counter();self.completed=[];self.rows=[];self.benchmarks=[]
        self.cache=None;self.previous=None;self.stages=int(os.environ.get('UNIFOLM_WORLD_FFN1280_ZERO_STAGES','2'))
        assert self.stages in(2,3)
        owner_module.zero_ffn1280_hidden_residual=self.observe;model.apply_model=self.apply

    def observe(self,inputs,parts,count,residual,bias=None):
        if self.step is not None and self.step[:2]==('world',0)and tuple(inputs[0].shape)==(2560,5120):
            if self.parts is None:
                assert not torch.cuda.is_current_stream_capturing()
                matches=[]
                for name,module in self.linears:
                    cache=getattr(module,'_decision_ffn_residual_weight_cache',None)
                    if cache is not None and cache[1][0]is parts[0]and cache[1][1]is parts[1]:matches.append((name,module))
                assert len(matches)==1
                self.name,module=matches[0];self.master=module.weight;self.master_version=self.master._version;self.parts=parts;self.count=count
            if self.parts[0]is parts[0]and torch.cuda.is_current_stream_capturing():
                assert self.inputs is None
                self.inputs=tuple(inputs)
        return self.original(inputs,parts,count,residual,bias)

    def baseline(self,inputs,output):
        _zero_low_gemm_residual[(triton.cdiv(2560,64)*20,)](*inputs,*self.parts,self.count,self.parts[0],self.zero,output,2560,1280,5120,False,64,32,1,num_warps=4,num_stages=self.stages,enable_fp_fusion=False)

    def run_candidate(self,inputs,output,flag):
        return candidate.run_device(inputs,self.parts,self.count,None,self.zero,output,self.cache,flag,bm=64,stages=self.stages)

    def benchmark(self,position):
        # Copies precede timing; both paths see identical actual adjacent inputs.
        current=tuple(t.clone()for t in self.inputs);old=self.previous
        first=torch.empty_like(self.zero);second=torch.empty_like(self.zero)
        def baseline():self.baseline(old,first);self.baseline(current,second)
        def delta():self.run_candidate(old,first,self.refresh_flag);self.run_candidate(current,second,self.reuse_flag)
        for _ in range(3):baseline();delta()
        torch.cuda.synchronize();graphs={}
        for name,call in(('baseline',baseline),('candidate',delta)):
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):call()
            torch.cuda.current_stream().wait_stream(stream);graphs[name]=graph
        pairs=[]
        for index in range(6):
            row={}
            for name in(('baseline','candidate')if index%2==0 else('candidate','baseline')):
                times=[]
                for _ in range(7):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(20):graphs[name].replay()
                    end.record();end.synchronize();times.append(begin.elapsed_time(end)/20.)
                row[name+'_ms']=statistics.median(times)
            row['saved_ms']=row['baseline_ms']-row['candidate_ms'];pairs.append(row)
        return dict(position=position,pairs=pairs,all_six_faster=all(p['saved_ms']>0 for p in pairs),median_saved_ms=statistics.median(p['saved_ms']for p in pairs),includes_refresh_pack_save_cache_and_reuse=True,matched_num_stages=self.stages,actual_shape=[2560,1280,5120],two_operations_per_measurement=True)

    def diagnose(self,position):
        assert self.inputs is not None and self.master._version==self.master_version
        if self.cache is None:
            self.zero=torch.zeros((2560,1280),device=self.inputs[0].device,dtype=torch.float32)
            self.reference=torch.empty_like(self.zero);self.output=torch.empty_like(self.zero)
            self.cache=candidate.allocate(self.inputs[0],1280)
            self.refresh_flag=torch.zeros((),device=self.zero.device,dtype=torch.int32)
            self.reuse_flag=torch.ones_like(self.refresh_flag)
            self.weight_low_nonzero=int(torch.count_nonzero(self.parts[1]));assert self.weight_low_nonzero==int(self.count)==0
        self.baseline(self.inputs,self.reference)
        row=dict(position=position,pure_prebias_product_reference=True)
        if position:
            self.run_candidate(self.inputs,self.output,self.reuse_flag)
            row['delta']=errors(self.reference,self.output)
            if position in(1,10,25,49):self.benchmarks.append(self.benchmark(position))
        self.run_candidate(self.inputs,self.output,self.refresh_flag)
        row['refresh_bitwise']=bool(torch.equal(self.reference.view(torch.int32),self.output.view(torch.int32)))
        assert row['refresh_bitwise']
        self.previous=tuple(t.clone()for t in self.inputs);self.rows.append(row)

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
            before=torch.cuda.get_rng_state().clone();self.diagnose(position)
            assert torch.equal(before,torch.cuda.get_rng_state())
        self.calls+=1;self.counts[stage]+=1;return result

    def end_step(self):
        assert self.step is not None and self.calls==1
        stage,generation,position=self.step
        if position==49:self.completed.append(dict(stage=stage,generation=generation,steps=50))
        self.previous_step=self.step;self.step=None

    def report(self):
        assert self.step is None and len(self.rows)==50 and len(self.benchmarks)==4
        return dict(diagnostic_only=True,read_only_temporal_delta_observation=True,normal_Torch_function_hook_only=True,no_native_argument_inspection=True,no_tensors_weights_or_addresses_exported=True,inference_outputs_unmodified=True,counts=dict(self.counts),completed_generations=self.completed,graph_count=len(self.graph.entries),owner=self.name,full_shape=[2560,1280,5120],weight_low_nonzero=self.weight_low_nonzero,rows=self.rows,benchmarks=self.benchmarks,RNG_unchanged=True,sampler_step_count_reduced=False,round_count_reduced=False,pure_prebias_error_denominator=True,production_quality_not_inferred=True,official_adopted=False)


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
    print('TEMPORAL_DELTA_OBSERVER_INSTALLED original_inference=true first_world50_only=true',flush=True)


def finish(args):
    report=active.report();assert report['counts']==dict(decision=50*args.n_iter,world=50*args.n_iter)and len(report['completed_generations'])==2*args.n_iter and report['graph_count']==2
    Path(args.profile_path).with_suffix('.crossstep.json').write_text(json.dumps(report,indent=2)+'\n')
    active.owner_module.zero_ffn1280_hidden_residual=active.original
    print('TEMPORAL_DELTA_OBSERVER_COMPLETE '+json.dumps(report),flush=True)
