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
from init_ffn219_mode import mode,measure,install,uninstall


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
                assert controller.report()['initial_FFN']['captured']==1
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
                changed=tuple(v+.0001 if isinstance(v,torch.Tensor) and v.is_floating_point() else v for v in values)
                check('changed_inputs',changed)
                masters={}
                for module in model.model.diffusion_model.modules():
                    if isinstance(module,torch.nn.Linear):
                        weight=module.weight
                        if weight.dtype==torch.float32:masters[weight.data_ptr()]=weight
                linear_pointers=set(masters)
                norms={p.data_ptr():p for module in model.model.diffusion_model.modules()
                    if isinstance(module,(torch.nn.GroupNorm,torch.nn.LayerNorm))
                    for p in module.parameters(recurse=False) if p.dtype==torch.float32}
                assert len(linear_pointers)==570 and len(norms)>148
                masters.update(norms)
                convolutions={p.data_ptr():p for module in model.model.diffusion_model.modules()
                    if isinstance(module,torch.nn.Conv2d)
                    for p in module.parameters(recurse=False) if p.dtype==torch.float32}
                assert len(convolutions)>40
                masters.update(convolutions)
                saved={ptr:w.detach().cpu().clone() for ptr,w in masters.items()}
                def conditioning_records():
                    records={}
                    for name,module in model.model.diffusion_model.named_modules():
                        for attribute in ['_prepared_context_cache','_global_condition_cache','_condition_projection_cache']:
                            value=getattr(module,attribute,None)
                            pairs=value.items() if attribute=='_condition_projection_cache' and isinstance(value,dict) else [(attribute,value)]
                            for label,cache in pairs:
                                if isinstance(cache,dict) and 'key' in cache and isinstance(cache.get('output'),torch.Tensor):
                                    records[(name,attribute,label)]=cache
                    return records
                def refresh(inputs,restored=False):
                    # Synthetic master edits are outside normal inference.
                    # Conditioning keys use input identities/versions, so a
                    # master-only edit must explicitly invalidate those keys.
                    before=None
                    if restored:
                        with mode(True):original.original(*inputs,**keywords)
                        with mode(False):prior=wrappers['baseline'](*inputs,**keywords)
                        prior_leaves,_=tree_flatten(prior)
                        # This intentionally stale baseline is an observation,
                        # not a candidate error gate. Zeroed conditioning can
                        # produce large errors until keys are refreshed.
                        before=[dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                            max_abs=float((a-b).abs().max()),
                            signedzero_bits=torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32)))
                            for a,b in zip(prior_leaves,baseline_reference)]
                    entries=conditioning_records();assert entries
                    pointers={key:cache['output'].data_ptr() for key,cache in entries.items()}
                    for cache in entries.values():cache['key']=None
                    with mode(True):original.original(*inputs,**keywords)
                    after=conditioning_records()
                    assert pointers=={key:cache['output'].data_ptr() for key,cache in after.items()}
                    if restored:
                        with mode(False):current=wrappers['baseline'](*inputs,**keywords)
                        current_leaves,_=tree_flatten(current)
                        post=[measure(a,b) for a,b in zip(current_leaves,baseline_reference)]
                        record=dict(restored=True,invalidated_keys=len(entries),original_output_storage_preserved=True,
                            before_explicit_invalidation=before,after_explicit_invalidation=post)
                        print('SYNTHETIC_MASTER_CONDITION_REFRESH '+json.dumps(record),flush=True)
                        assert all(p['signedzero_bits'] for p in post),record
                    else:record=dict(restored=False,invalidated_keys=len(entries),original_output_storage_preserved=True)
                    condition_refresh_records.append(record)
                for weight in masters.values():weight.add_(.00013)
                refresh(changed,False);check('changed_all_masters',changed)
                for ptr,weight in masters.items():weight.copy_(saved[ptr])
                refresh(values,True);check('restored_all_masters',values)
                for weight in masters.values():weight.zero_()
                refresh(values,False);check('zero_all_masters',values)
                for ptr,weight in masters.items():weight.copy_(saved[ptr])
                refresh(values,True);check('zero_then_restore',values)
                assert torch.equal(rng,torch.cuda.get_rng_state())
                del saved
                for index in range(40):wrappers['candidate' if index%2 else 'baseline'](*values,**keywords)
                torch.cuda.synchronize();pairs=[]
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for index in range(6):
                    pair={}
                    for label in (['baseline','candidate'] if index%2==0 else ['candidate','baseline']):
                        dist.barrier();torch.cuda.synchronize()
                        fn=wrappers[label]
                        for _ in range(2):fn(*values,**keywords)
                        torch.cuda.synchronize();samples=[]
                        for _ in range(7):
                            start.record();fn(*values,**keywords);end.record();end.synchronize()
                            samples.append(start.elapsed_time(end))
                        pair[label+'_samples_ms']=samples;pair[label+'_ms']=statistics.median(samples)
                    pair['saved_ms']=pair['baseline_ms']-pair['candidate_ms'];pairs.append(pair)
                entries={k:next(iter(w.entries.values())) for k,w in wrappers.items()}
                reports={k:entry['numerical_sgemm'].report if entry.get('numerical_sgemm') is not None else entry['native_weight_cache'].report for k,entry in entries.items()}
                summary=dict(stage=options.profile_stage,cases=cases,pairs=pairs,
                    median_saved_ms=statistics.median(p['saved_ms'] for p in pairs),
                    same_process=True,same_GPU_pair=True,same_model_masters=True,
                    current209_norm2_presplit_flags_preserved=True,
                    original_graph_owners_and_all_version_checks_both_modes=True,
                    initial_FFN_row_sharding_on_adopted219=True,baseline_is_adopted_parallel_routing=True,all_diffusion_all_masters_tested=True,attention_launches=controller.report(),baseline_restoration_bitwise=True,synthetic_master_condition_refresh=condition_refresh_records,full_video_PSNR_unverified=True,
                    balanced_warmup=40,settle_per_mode=2,samples_per_condition=7,
                    Conv2d_master_tensors=len(convolutions),Linear_master_tensors=len(linear_pointers),norm_master_tensors=len(norms),master_refresh_uses_existing_eager_forward=True,
                    RNG_unchanged=True,reports=reports,no_official_score=True)
                (options.diagnostic_dir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
                print('NCCL_GEGLU_209_MODEL '+json.dumps(summary),flush=True)
                print('NCCL_GEGLU_209_MODEL_CHECK passed=true stage='+options.profile_stage+' six_cases_gross=true baseline_restored_bitwise=true no_official_score=true',flush=True)
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
