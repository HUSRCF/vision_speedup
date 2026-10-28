"""Full-denoiser paired graph timing on real Z1 workload inputs, one GPU."""
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import torch
from torch.utils._pytree import tree_flatten,tree_map
from unifolm_wma.utils import inference_runtime as runtime
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from nonv2_fused_half_geglu_mode import install,uninstall,mode


class DiagnosticComplete(Exception):pass


def error(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and torch.isfinite(a).all() and torch.isfinite(b).all()
    maximum=float((a.float()-b.float()).abs().max())
    relative=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30))
    exact=torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    return dict(signedzero_bits=exact,max_abs=maximum,relative_l2=relative,gross_passed=relative<=.005 or maximum<=.002)


def main():
    stage=os.environ['EXPERIMENT_MODEL_STAGE'];assert stage in ('decision','world')
    previous=runtime.configure_denoiser
    def configure(model,*args,**kwargs):
        previous(model,*args,**kwargs);original=model.apply_model;assert isinstance(original,DenoiserGraph)
        controller=install(model.model.diffusion_model);calls=0
        def diagnose(*values,**keywords):
            nonlocal calls
            calls+=1;target=1 if stage=='decision' else 51
            if calls<target:return original(*values,**keywords)
            assert calls==target
            values,keywords=tree_map(lambda x:x.detach().clone().contiguous()if isinstance(x,torch.Tensor)else x,(values,keywords))
            wrappers={};saved={};masters={}
            try:
                for label,on in [('baseline',False),('candidate',True)]:
                    wrapper=DenoiserGraph(original.original,condition_refresher=original.condition_refresher,skip_unchanged_inputs=False)
                    wrappers[label]=wrapper
                    with mode(on):wrapper(*values,**keywords)
                    assert len(wrapper.entries)==1
                ctl=controller.report();assert ctl['captured']>0
                assert len({x['module']for x in ctl['rows']})==ctl['captured']
                names={x['module']for x in ctl['rows']}
                for name,module in model.model.diffusion_model.named_modules():
                    if name in names:
                        for p in module.proj.parameters():masters[p.data_ptr()]=p
                assert len(masters)==2*ctl['captured']
                saved={ptr:p.detach().cpu().clone()for ptr,p in masters.items()}
                rng=torch.cuda.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone();cases=[];reference=None
                def check(label,inputs):
                    nonlocal reference
                    with mode(False):left=wrappers['baseline'](*inputs,**keywords)
                    with mode(True):right=wrappers['candidate'](*inputs,**keywords)
                    aa,sa=tree_flatten(left);bb,sb=tree_flatten(right);assert sa==sb and len(aa)==3
                    errors=[error(b,a)for a,b in zip(aa,bb)]
                    assert all(e['gross_passed']for e in errors),(label,errors)
                    if label=='initial':reference=[a.clone()for a in aa]
                    if label in ('restored_masters','zero_then_restore'):
                        assert all(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))for a,b in zip(aa,reference))
                    if label=='zero_masters':
                        assert any(not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))for a,b in zip(aa,reference))
                    cases.append(dict(case=label,errors=errors))
                check('initial',values)
                changed=tree_map(lambda x:x+.001 if isinstance(x,torch.Tensor)and x.is_floating_point()else x,values)
                check('changed_inputs',changed)
                for p in masters.values():p.add_(.00013)
                check('changed_masters',changed)
                for ptr,p in masters.items():p.copy_(saved[ptr])
                check('restored_masters',values)
                for p in masters.values():p.zero_()
                check('zero_masters',values)
                for ptr,p in masters.items():p.copy_(saved[ptr])
                check('zero_then_restore',values);saved.clear()
                assert torch.equal(rng,torch.cuda.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state())
                for i in range(40):wrappers['candidate'if i%2 else'baseline'](*values,**keywords)
                torch.cuda.synchronize();pairs=[]
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for i in range(6):
                    pair={}
                    for label in (('baseline','candidate')if i%2==0 else('candidate','baseline')):
                        for _ in range(2):wrappers[label](*values,**keywords)
                        samples=[]
                        for _ in range(7):
                            start.record();wrappers[label](*values,**keywords);end.record();end.synchronize();samples.append(start.elapsed_time(end))
                        pair[label+'_ms']=statistics.median(samples);pair[label+'_samples_ms']=samples
                    pair['saved_ms']=pair['baseline_ms']-pair['candidate_ms'];pairs.append(pair)
                report=dict(passed=True,stage=stage,cases=cases,pairs=pairs,controller=ctl,
                    median_saved_ms=statistics.median(p['saved_ms']for p in pairs),all6_pairs_positive=all(p['saved_ms']>0 for p in pairs),
                    all_three_outputs_six_states_bitwise=all(e['signedzero_bits']for c in cases for e in c['errors']),
                    RNG_unchanged=True,baseline_restoration_bitwise=True,mutated_projection_tensors=len(masters),
                    original59_nonv2_baseline=True,no_full_video_quality_or_process_time=True,no_official_score=True)
                Path(os.environ['EXPERIMENT_REPORT']).write_text(json.dumps(report,indent=2)+'\n')
                print('NONV2_FUSED_GEGLU_MODEL passed=true stage='+stage,flush=True)
            finally:
                for ptr,p in masters.items():
                    if ptr in saved:p.copy_(saved[ptr])
                for wrapper in [original,*wrappers.values()]:
                    for entry in wrapper.entries.values():entry['graph'].reset()
                    wrapper.entries.clear()
            raise DiagnosticComplete
        model.apply_model=diagnose
    runtime.configure_denoiser=configure
    base=Path(os.environ['MODEL_BASE_ROOT']);entry=base/'scripts/evaluation/world_model_interaction.py'
    sys.path.insert(0,str(entry.parent));sys.argv=[str(entry),*json.loads(os.environ['EXPERIMENT_INFERENCE_ARGS'])]
    try:runpy.run_path(str(entry),run_name='__main__')
    except DiagnosticComplete:pass
    finally:runtime.configure_denoiser=previous;uninstall()


if __name__=='__main__':main()
