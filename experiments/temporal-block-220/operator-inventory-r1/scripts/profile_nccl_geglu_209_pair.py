"""Paired adopted219 graphs; only world initial FFN row routing differs."""
import argparse
import json
from pathlib import Path
import runpy
import statistics
import sys
import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch.utils._pytree import tree_flatten
import unifolm_wma.utils.inference_runtime as runtime
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from temporal220_observer import mode,measure,install,uninstall


class DiagnosticComplete(Exception):pass


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--diagnostic-dir',type=Path,required=True)
    p.add_argument('--profile-stage',choices=['decision','world'],required=True)
    options,inference_args=p.parse_known_args()
    options.diagnostic_dir.mkdir(parents=True,exist_ok=False)
    previous_configure=runtime.configure_denoiser
    def configure(model,*a,**k):
        previous_configure(model,*a,**k);original=model.apply_model
        controller=install(model.model.diffusion_model)
        assert isinstance(original,DenoiserGraph)
        def diagnose(*values,**keywords):
            if torch.backends.cudnn.allow_tf32!=(options.profile_stage=='decision'):
                return original(*values,**keywords)
            wrappers={}
            try:
                for label,on in [('baseline',False),('candidate',True)]:
                    wrapper=DenoiserGraph(original.original,condition_refresher=original.condition_refresher,skip_unchanged_inputs=False)
                    wrappers[label]=wrapper
                    dist.barrier();torch.cuda.synchronize()
                    with mode(on):wrapper(*values,**keywords)
                    assert len(wrapper.entries)==1
                assert controller.report()['captured']==17
                for wrapper in wrappers.values():
                    entry=next(iter(wrapper.entries.values()))
                    if options.profile_stage=='world':
                        owner=entry['numerical_sgemm']
                        assert len(owner.swaps)==23 and owner.report['numerical_selected']==15 and owner.report['kahan_selected']==8
                        assert len(owner._world_head_filter_cache.caches)==45
                    else:
                        owner=entry['native_weight_cache']
                        assert len(owner.caches)==138 and len(owner._native_conv1d_weight_cache.caches)==43 and len(owner._native_remaining_filter_cache.caches)==8
                rng=torch.cuda.get_rng_state();cases=[];baseline_reference=None;condition_refresh_records=[]
                def check(label,inputs):
                    nonlocal baseline_reference
                    with mode(False):left=wrappers['baseline'](*inputs,**keywords)
                    with mode(True):right=wrappers['candidate'](*inputs,**keywords)
                    aa,sa=tree_flatten(left);bb,sb=tree_flatten(right);assert sa==sb and len(aa)==3
                    errors=[measure(b,a) for a,b in zip(aa,bb)]
                    if label=='initial':baseline_reference=[a.clone() for a in aa]
                    restored=label in {'restored_all_masters','zero_then_restore'}
                    if restored:
                        assert all(torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32)) for a,b in zip(aa,baseline_reference))
                    cases.append(dict(case=label,three_outputs_gross_passed=True,errors=errors,
                        baseline_restored_bitwise=restored,attention_launches=controller.report()))
                check('initial',values)
                assert torch.equal(rng,torch.cuda.get_rng_state())
                summary=dict(stage=options.profile_stage,cases=cases,pairs=[],
                    attention_launches=controller.report(),RNG_unchanged=True,
                    baseline_is_exact_adopted220=True,read_only_operator_inventory=True,
                    all_diffusion_all_masters_tested=False,no_official_score=True,
                    no_speed_or_video_qualification=True)
                (options.diagnostic_dir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
                print('NCCL_GEGLU_209_MODEL '+json.dumps(summary),flush=True)
                print('NCCL_GEGLU_209_MODEL_CHECK passed=true stage='+options.profile_stage+' one_case_bitwise=true read_only_inventory=true no_official_score=true',flush=True)
            finally:
                for wrapper in [original,*wrappers.values()]:
                    for entry in wrapper.entries.values():
                        for key in ['numerical_sgemm','native_weight_cache']:
                            if entry.get(key) is not None:entry[key].close()
                        entry['graph'].reset()
                    wrapper.entries.clear()
            raise DiagnosticComplete
        model.apply_model=diagnose
    runtime.configure_denoiser=configure
    script=Path(__file__).parent/'evaluation/world_model_interaction.py';sys.path.insert(0,str(script.parent.resolve()));sys.argv=[str(script),*inference_args]
    try:runpy.run_path(str(script),run_name='__main__')
    except DiagnosticComplete:pass
    finally:
        runtime.configure_denoiser=previous_configure
        uninstall()


if __name__=='__main__':main()
