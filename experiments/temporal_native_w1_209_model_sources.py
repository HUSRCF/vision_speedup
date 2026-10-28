"""Both real-stage graphs, all Linear/norm masters and exact three-output checks."""
import ast,json
from pathlib import Path
from attention_protected_scores_model_sources import sources as previous_sources

def sources(configs):
    old=previous_sources('');profile=old['scripts/profile_attention_protected_scores_pair.py']
    profile=profile.replace('import torch\n','import torch\nimport torch.utils.checkpoint\n',1)
    profile=profile.replace('attention_protected_scores_mode','temporal_native_w1_209_mode').replace('attention-protected-scores-selection.json','temporal-native-w1-209-selection.json')
    profile=profile.replace('ATTENTION_PROTECTED_SCORES','TEMPORAL_NATIVE_W1_209').replace('attention_protected_scores=True','temporal_native_w1=True').replace('current188','current209')
    anchor='                assert len(masters)>=4 and all(w.dtype==torch.float32 for w in masters.values())\n';assert profile.count(anchor)==1
    profile=profile.replace(anchor,'''                linear_pointers=set(masters)
                norms={p.data_ptr():p for module in model.model.diffusion_model.modules()
                    if isinstance(module,(torch.nn.GroupNorm,torch.nn.LayerNorm))
                    for p in module.parameters(recurse=False) if p.dtype==torch.float32}
                assert len(linear_pointers)==570 and len(norms)>148
                masters.update(norms)
''')
    profile=profile.replace('Linear_master_tensors=len(masters),','Linear_master_tensors=len(linear_pointers),norm_master_tensors=len(norms),')
    anchor='                rng=torch.cuda.get_rng_state();cases=[];';assert profile.count(anchor)==1
    profile=profile.replace(anchor,'''                assert controller.report()['captured']==34
                for wrapper in wrappers.values():
                    entry=next(iter(wrapper.entries.values()))
                    if options.profile_stage=='world':
                        owner=entry['numerical_sgemm']
                        assert len(owner.swaps)==23 and owner.report['numerical_selected']==15 and owner.report['kahan_selected']==8
                        assert len(owner._world_head_filter_cache.caches)==45
                    else:
                        owner=entry['native_weight_cache']
                        assert len(owner.caches)==138 and len(owner._native_conv1d_weight_cache.caches)==43 and len(owner._native_remaining_filter_cache.caches)==8
'''+anchor)
    runner=old['scripts/run_attention_protected_scores_pairs.py']
    runner=runner.replace('PAIR_PROFILE_SOURCE_RUN','KERNEL_PROFILE_SOURCE_RUN')
    begin=runner.index('flags=dict(');end=runner.index("assert flags['UNIFOLM_ATTENTION_OUTPUT_NOFILL']",begin)
    runner=runner[:begin]+"flags=json.loads(Path(os.environ['CURRENT_EXPERIMENT_FLAGS_FILE']).read_text())\n"+runner[end:]
    runner=runner.replace('attention-protected-scores-current-pairs-','temporal-native-w1-209-model-').replace('profile_attention_protected_scores_pair.py','profile_temporal_native_w1_209_pair.py')
    begin=runner.index("with (output/'guard.log')");end=runner.index('rows={}\n',begin)
    runner=runner[:begin]+'''with (output/'guard.log').open('x') as log:subprocess.run([sys.executable,'scripts/check_temporal_native_w1_209_cpu.py'],env=dict(environment,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
assert 'TEMPORAL_NATIVE_W1_209_CPU_CHECK passed=true' in (output/'guard.log').read_text()
'''+runner[end:]
    runner=runner.replace('ATTENTION_PROTECTED_SCORES','TEMPORAL_NATIVE_W1_209')
    runner=runner.replace("rows[stage]=json.loads((directory/'comparison.json').read_text());print", "rows[stage]=json.loads((directory/'comparison.json').read_text());assert all(e['signedzero_bits'] for c in rows[stage]['cases'] for e in c['errors']);assert rows[stage]['attention_launches']['captured']==34;print")
    runner=runner.replace('dict(stages=rows,no_official_score=True)', "dict(stages=rows,ready_for_full_trial=all(p['saved_ms']>0 for r in rows.values() for p in r['pairs']),no_official_score=True)")
    files={'scripts/profile_temporal_native_w1_209_pair.py':profile,'scripts/run_temporal_native_w1_209_pairs.py':runner,
        'temporal-native-w1-209-selection.json':json.dumps(configs,indent=2)+'\n'}
    for name in ('temporal_native_w1_209_mode.py','check_temporal_native_w1_209_cpu.py'):files['scripts/'+name]=Path(__file__).with_name(name).read_text()
    for name,text in files.items():
        if name.endswith('.py'):ast.parse(text)
    return files
