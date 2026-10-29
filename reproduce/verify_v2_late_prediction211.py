"""Verify late V2 source/CPU identity independently from full video quality."""
import ast
import hashlib
import json
from pathlib import Path


def verify(label='both-late-r3', cpu_label='public-late-repro-cpu-r3', input_start=6, output_end=6):
    repo=Path(__file__).resolve().parents[1]
    load=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    frozen=repo/'experiments/v2-prediction211'/label
    overlay=frozen/'production-overlay'
    base=repo/'unifolm'
    evidence=repo/'results/aggressive-20261029/v2-prediction211'
    original=load(base/'world-head-filter-cache-production-sources.json')
    manifest=load(overlay/'crossstep211-production-sources.json')
    feature='src/unifolm_wma/utils/crossstep_feature_reuse.py'
    runtime='src/unifolm_wma/utils/crossstep_feature_runtime.py'
    graph='src/unifolm_wma/utils/cuda_graph.py'
    assert len(manifest)==211 and set(manifest)-set(original)=={feature,runtime}
    assert {n for n in original if original[n]!=manifest[n]}=={graph}
    for name,digest in manifest.items():
        p=overlay/name
        assert sha(p if p.exists()else base/name)==digest,name
    for name,digest in load(frozen/'asset-hashes.json').items():assert sha(overlay/name)==digest,name
    for name,digest in load(frozen/'guard-source-hashes.json').items():assert sha(frozen/'guard'/name)==digest,name
    previous=repo/'experiments/v2-prediction211/both-r2/production-overlay'
    for name in(feature,graph):assert sha(overlay/name)==sha(previous/name)
    entry='scripts/evaluation/world_model_interaction_crossstep.py'
    assert (overlay/entry).read_text()==(previous/entry).read_text().replace("'start': 10, 'stop': 40","'start': 20, 'stop': 40").replace("'input_start': 6, 'output_end': 6",f"'input_start': {input_start}, 'output_end': {output_end}")
    text=(overlay/runtime).read_text()
    tree=ast.parse(text)
    scope=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_weight_cache_scope')
    scope=ast.get_source_segment(text,scope)
    assert "flags=('UNIFOLM_NATIVE_CONV1D_WEIGHT_CACHE','UNIFOLM_NATIVE_REMAINING_FILTER_CACHE')"in scope
    assert "assert os.environ.get('UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE')=='1'"in scope
    assert "os.environ['UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE']"not in scope
    assert 'finally:'in scope and 'os.environ.pop(name,None)'in scope
    trial=load(frozen/'trial.json')
    assert (trial['rounds'],trial['videos'],trial['source_count'])==(11,23,211)
    assert trial['feature_policy']==dict(input_start=input_start,output_end=output_end,start=20,stop=40,
        boundary_residual=False,scope='deep',stages=['decision','world'])
    assert trial['generic_decision_weight_cache_preserved']and trial['specialized_cache_flags_scoped']
    assert trial['prediction_policy']['all_rounds_and50_steps_preserved']and not trial['official_adopted']
    flags=load(frozen/'guard/flags.json')
    assert flags['UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE']=='1'and flags['UNIFOLM_ATTENTION_DTYPE']=='fp32'
    cpu_root=evidence/cpu_label
    cpu=load(cpu_root/'summary.json')
    assert cpu['passed']and cpu['no_GPU_execution']
    assert cpu['public_script_SHA256']==sha(repo/'reproduce/run_v2_prediction211.py')
    assert len(cpu['trials'])==1
    row=cpu['trials'][0]
    assert row['trial']==label and row['source_count']==211
    for name in('exact_timed_manifest_and_assets','predictor_affine3600_CPU_passed',
                 'stage_independent_of_TF32','generic_cache_preserved',
                 'specialized_cache_scope_exception_restore','late_window1100_calls220_predictions'):
        assert row[name],name
    proof=load(cpu_root/'late-window-source-proof.json')
    assert proof['late_window_script_SHA256']==sha(frozen/'check_late_window_cpu.py')
    for name,digest in proof['logs_SHA256'].items():assert sha(cpu_root/name)==digest
    full=evidence/label
    summary=full/'summary.json'
    if summary.exists():
        result=load(summary)
        assert not result['official_adopted']and result['original_rounds_and_sampler_updates_preserved']
        if result['clean']:
            assert result['workload_complete']
            run=full/'run'
            row=next(r for r in load(run/'result.json')['cases']if r['case']==trial['case'])
            assert (row['rounds'],row['videos'])==(11,23)
            assert row['process_wall_s']==result['process_wall_s']and row['psnr_db']==result['psnr_db']
            assert row['psnr_passed']==result['quality_passed']
            expected={'both-late-r3':(310.127282,18.889815362652385),'both-late7-r4':(325.592034,20.224752075187556)}[label]
            assert (row['process_wall_s'],row['psnr_db'])==expected and not row['psnr_passed']
            feature_report=load(run/'profiles/unitree_z1_dual_arm_stackbox_v2.crossstep.json')
            assert feature_report['graph_count']==4 and feature_report['window']==[20,40]
            assert len(feature_report['selected_blocks'])==13-input_start+output_end
            assert feature_report['counts']==dict(decision_calls=550,decision_refresh=440,decision_reuse=110,
                                                   world_calls=550,world_refresh=440,world_reuse=110)
            assert feature_report['alpha_histogram']=={'1.0':22,'0.5':198}
            assert len(feature_report['completed_generations'])==22
            assert all(r['steps']==50 for r in feature_report['completed_generations'])
            assert feature_report['decision_reuse_native_weight_cache']
            assert not feature_report['decision_reuse_specialized_filter_caches']
            assert feature_report['graph_prediction_selfcheck']['passed']
            assert not feature_report['sampler_step_count_reduced']and not feature_report['round_count_reduced']
    formal=load(repo/'results/result.json')
    assert next(r for r in formal['cases']if r['case']==trial['case'])['process_wall_s']==257.945205
    print('V2_LATE_PREDICTION211_VERIFIED exact211=true generic_cache_retained=true specialized_scope_checked=true CPU_quality_not_inferred=true not_adopted=true')


if __name__=='__main__':
    verify()
    verify('both-late7-r4','public-late7-repro-cpu-r4',9,3)
