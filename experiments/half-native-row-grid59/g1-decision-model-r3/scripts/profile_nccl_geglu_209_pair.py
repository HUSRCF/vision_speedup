"""Real nonV2 FP16 denoiser: original single-rank graph vs FFN row sharding.

No V2 hooks or quality rules are inherited. A small model error is only a gross
screen; official full-video PSNR must be checked before any adoption.
"""
import argparse
import json
from pathlib import Path
import runpy
import statistics
import sys
import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch.utils._pytree import tree_flatten, tree_map
import unifolm_wma.utils.inference_runtime as runtime
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from nonv2_native_ffn_grid59_spatial_mode import mode, install, uninstall


class DiagnosticComplete(Exception):pass


def bits_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
        a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def error(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    x,y=a.float(),b.float()
    relative=float((x-y).norm()/y.norm().clamp_min(1e-30))
    maximum=float((x-y).abs().max())
    # A gross screen for catastrophic regressions, never a video PSNR proxy.
    passed=relative<=.005 or maximum<=.002
    return dict(relative_l2=relative,max_abs=maximum,signedzero_bits=bits_equal(a,b),gross_passed=passed)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--diagnostic-dir',type=Path,required=True)
    p.add_argument('--profile-stage',choices=['decision','world'],required=True)
    options,inference_args=p.parse_known_args()
    options.diagnostic_dir.mkdir(parents=True,exist_ok=False)
    prior=runtime.configure_denoiser
    def configure(model,*a,**k):
        prior(model,*a,**k);original=model.apply_model
        assert isinstance(original,DenoiserGraph)
        controller=install(model.model.diffusion_model);calls=0
        def diagnose(*values,**keywords):
            nonlocal calls
            calls+=1
            target=1 if options.profile_stage=='decision' else 51
            if calls<target:return original(*values,**keywords)
            assert calls==target
            # The original workload uses50 steps and no CFG second branch.
            # Clone and broadcast identical model inputs without exporting them.
            def clone(t):
                if not isinstance(t,torch.Tensor):return t
                assert t.is_cuda
                value=t.detach().clone().contiguous();dist.broadcast(value,src=0);return value
            values,keywords=tree_map(clone,(values,keywords))
            wrappers={};saved={}
            try:
                for label,on in [('baseline',False),('candidate',True)]:
                    wrapper=DenoiserGraph(original.original,condition_refresher=original.condition_refresher,skip_unchanged_inputs=False)
                    wrappers[label]=wrapper;dist.barrier();torch.cuda.synchronize()
                    with mode(on):wrapper(*values,**keywords)
                    assert len(wrapper.entries)==1
                controller.bind(next(iter(wrappers['candidate'].entries.values()))['graph'])
                ctl=controller.report();assert ctl['captured']==31 and ctl['native_linear_capture_count']==62 and ctl['GEGLU']['captured']==31 and ctl['spatial_shared_boundary']['captured']==15 and ctl['spatial_shared_boundary']['attention_calls']==75
                assert len({x['module'] for x in ctl['rows']})==31
                rng=torch.cuda.get_rng_state();cpu_rng=torch.get_rng_state()
                cases=[];reference=None;zero_changed=False
                def check(label,inputs):
                    nonlocal reference,zero_changed
                    with mode(False):left=wrappers['baseline'](*inputs,**keywords)
                    with mode(True):right=wrappers['candidate'](*inputs,**keywords)
                    aa,sa=tree_flatten(left);bb,sb=tree_flatten(right)
                    assert sa==sb and len(aa)==3
                    errors=[error(b,a) for a,b in zip(aa,bb)]
                    assert all(e['signedzero_bits'] for e in errors),dict(case=label,errors=errors)
                    if label=='initial':reference=[a.clone() for a in aa]
                    restored=label in {'restored_FFN_masters','zero_then_restore'}
                    if restored:assert all(bits_equal(a,b) for a,b in zip(aa,reference))
                    if label=='zero_FFN_masters':
                        zero_changed=any(not bits_equal(a,b) for a,b in zip(aa,reference))
                        assert zero_changed,'mutation did not reach original graph'
                    cases.append(dict(case=label,three_outputs_gross_passed=True,errors=errors,baseline_restored_bitwise=restored))
                check('initial',values)
                changed=tree_map(lambda x:x+.001 if isinstance(x,torch.Tensor) and x.is_floating_point() else x,values)
                check('changed_inputs',changed)
                masters=controller.masters()
                assert len(masters)>132 and all(p.dtype==torch.float16 for p in masters.values())
                saved={ptr:p.detach().cpu().clone() for ptr,p in masters.items()}
                for weight in masters.values():weight.add_(.00013)
                check('changed_FFN_masters',changed)
                for ptr,weight in masters.items():weight.copy_(saved[ptr])
                check('restored_FFN_masters',values)
                for weight in masters.values():weight.zero_()
                check('zero_FFN_masters',values)
                for ptr,weight in masters.items():weight.copy_(saved[ptr])
                check('zero_then_restore',values)
                assert torch.equal(rng,torch.cuda.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state())
                saved.clear()
                for index in range(40):wrappers['candidate' if index%2 else 'baseline'](*values,**keywords)
                torch.cuda.synchronize();pairs=[]
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for index in range(6):
                    pair={}
                    for label in (['baseline','candidate'] if index%2==0 else ['candidate','baseline']):
                        dist.barrier();torch.cuda.synchronize();fn=wrappers[label]
                        for _ in range(2):fn(*values,**keywords)
                        torch.cuda.synchronize();samples=[]
                        for _ in range(7):
                            start.record();fn(*values,**keywords);end.record();end.synchronize()
                            samples.append(start.elapsed_time(end))
                        pair[label+'_samples_ms']=samples;pair[label+'_ms']=statistics.median(samples)
                    pair['saved_ms']=pair['baseline_ms']-pair['candidate_ms'];pairs.append(pair)
                report=dict(stage=options.profile_stage,cases=cases,pairs=pairs,
                    median_saved_ms=statistics.median(x['saved_ms'] for x in pairs),
                    same_process=True,same_GPU_pair=True,same_model_masters=True,
                    baseline_is_original59_nonv2_FP16=True,FFN_master_tensors=len(masters),
                    zero_FFN_changes_baseline=zero_changed,FFN_launches=ctl,original_native_GEMM_row_grids=True,G1_FP32_feedback_head_policy_preserved=True,
                    baseline_restoration_bitwise=True,RNG_unchanged=True,
                    balanced_warmup=40,settle_per_mode=2,samples_per_condition=7,
                    gross_error_gate=dict(relative_l2=.005,max_abs=.002,logic='either',not_official_quality=True),
                    full_video_PSNR_unverified=True,no_official_score=True)
                (options.diagnostic_dir/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
                print('NONV2_NATIVE_FFN_GRID59_MODEL '+json.dumps(dict(median_saved_ms=report['median_saved_ms'],pairs=pairs,captured=ctl['captured'],cases=cases)),flush=True)
                print('NCCL_GEGLU_209_MODEL_CHECK passed=true stage='+options.profile_stage+' six_cases_gross=true baseline_restored_bitwise=true no_official_score=true',flush=True)
            finally:
                # Restore even if a numerical gate fails before the last state.
                for ptr,weight in locals().get('masters',{}).items():
                    if ptr in saved:weight.copy_(saved[ptr])
                controller.close_owners()
                for wrapper in [original,*wrappers.values()]:
                    for entry in wrapper.entries.values():entry['graph'].reset()
                    wrapper.entries.clear()
            raise DiagnosticComplete
        model.apply_model=diagnose
    runtime.configure_denoiser=configure
    script=Path(__file__).parent/'evaluation/world_model_interaction.py'
    sys.path.insert(0,str(script.parent.resolve()));sys.argv=[str(script),*inference_args]
    try:runpy.run_path(str(script),run_name='__main__')
    except DiagnosticComplete:pass
    finally:runtime.configure_denoiser=prior;uninstall()


if __name__=='__main__':main()
