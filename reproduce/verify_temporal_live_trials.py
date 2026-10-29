"""Verify actual-operand errors, live-delta failure and complete Z1 reuse trials."""
import hashlib,json,statistics
from pathlib import Path


def verify():
    repo=Path(__file__).resolve().parents[1];code=repo/'experiments/temporal-live-trials';evidence=repo/'results/aggressive-20261029/temporal-live-trials'
    load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in sorted(code.iterdir()):
        metadata=load(folder/'trial.json');base=repo/metadata['base'];overlay=folder/'production-overlay'
        manifest=load(overlay/'crossstep211-production-sources.json');assert len(manifest)==metadata['source_count']
        nonv2=metadata['source_count']==61
        original=load(base/('nonv2-aggressive-production-sources.json'if nonv2 else'world-head-filter-cache-production-sources.json'))
        changed={n for n,h in original.items()if manifest[n]!=h}
        assert changed==({'src/unifolm_wma/utils/cuda_graph.py'}if nonv2 else set())
        for n,h in manifest.items():
            p=overlay/n;assert sha(p if p.exists()else base/n)==h,n
        for n,h in load(folder/'asset-hashes.json').items():assert sha(overlay/n)==h,n
        for n,h in load(folder/'guard-source-hashes.json').items():assert sha(folder/'guard'/n)==h,n
        if nonv2:assert load(folder/'guard/flags.json')==load(base.parent/(metadata['case']+'-flags.json'))
    actual_code=code/'actual-input-r1';delta_code=code/'delta-full-r2'
    assert load(actual_code/'guard/flags.json')==load(delta_code/'guard/flags.json')
    candidate='production-overlay/src/unifolm_wma/utils/temporal_delta_gemm_candidate.py'
    qualified=repo/'experiments/temporal-delta-gemm/primitive-r3/scripts/temporal_delta_gemm_candidate.py'
    if not qualified.exists():qualified=qualified.parent.parent/'temporal_delta_gemm_candidate.py'
    assert sha(actual_code/candidate)==sha(delta_code/candidate)==sha(qualified)
    actual=load(evidence/'actual-input-r1/guard-report.json')['diagnostic'];r=actual['crossstep']
    assert r['read_only_temporal_delta_observation']and r['pure_prebias_error_denominator']and r['inference_outputs_unmodified']
    assert r['counts']==dict(decision=550,world=550)and r['graph_count']==2 and len(r['completed_generations'])==22
    assert len(r['rows'])==50 and all(row['refresh_bitwise']for row in r['rows'])and r['weight_low_nonzero']==0
    assert [b['position']for b in r['benchmarks']]==[1,10,25,49]
    for b in r['benchmarks']:
        assert b['includes_refresh_pack_save_cache_and_reuse']and b['matched_num_stages']==3 and b['all_six_faster']
        assert len(b['pairs'])==6 and all(p['saved_ms']>0 for p in b['pairs'])
        assert b['median_saved_ms']==statistics.median(p['saved_ms']for p in b['pairs'])
    assert all(row['delta']['finite']for row in r['rows'][1:])
    hashes=load(evidence/'actual-input-r1/baseline-video-comparison.json')
    assert hashes['all23_SHA256_equal']and len(hashes['current_hashes'])==23 and hashes['current_hashes']==hashes['baseline_hashes']
    assert actual['case_result']['psnr_db']==29.675324786148643
    for required in('delta-full-r2','z1-both-r2','z1-world-r2'):assert(evidence/required/'summary.json').is_file()
    for label in('delta-full-r2','z1-both-r2','z1-world-r2','z1-both-r3','z1-both-r4'):
        out=evidence/label
        if not(out/'summary.json').exists():continue # Prepared trials have no performance/quality claim.
        summary=load(out/'summary.json');assert not summary['official_adopted']
        if not summary['clean']:
            assert label in('z1-both-r3','z1-both-r4')and summary['process_wall_s']is None and not summary['quality_passed']and not summary['workload_complete']
            assert(out/'guard.log').is_file();continue
        assert summary['workload_complete']
        report=load(out/'guard-report.json')['diagnostic'];reuse=report['crossstep'];metadata=load(code/label/'trial.json');case=metadata['case'];rounds=metadata['rounds']
        row=next(c for c in load(out/'run/result.json')['cases']if c['case']==case)
        assert row==report['case_result']and row['rounds']==rounds and row['videos']==2*rounds+1
        assert row['execution_ok']and row['profile_complete']and row['process_wall_s']==summary['process_wall_s']and row['psnr_db']==summary['psnr_db']
        assert row['psnr_passed']==summary['quality_passed']==(row['psnr_db']>25)
        assert reuse['counts']['decision_calls']==reuse['counts']['world_calls']==50*rounds
        assert len(reuse['completed_generations'])==2*rounds and all(g['steps']==50 for g in reuse['completed_generations'])
        assert not reuse['sampler_step_count_reduced']and not reuse['round_count_reduced']
        profile=load(out/'run/profiles'/f'{case}.json');assert profile['total_process_wall_s']==row['process_wall_s']
        if label=='delta-full-r2':
            assert reuse['owner_count']==12 and reuse['counts']['world_reuse']==165 and reuse['graph_count']==2
            assert reuse['counts'].get('decision_reuse',0)==0 and all(o['captures']==1 for o in reuse['owners'])
            assert reuse['fresh_current_inputs_bias_and_residual']and reuse['condition_and_generation_invalidation']
            assert not row['psnr_passed']and row['psnr_db']==24.82881782867743
        else:
            policy=metadata['feature_policy'];assert len(reuse['selected_blocks'])==12-policy['input_start']+1+policy['output_end']
            assert reuse['counts']['world_reuse']==180 and reuse['graph_count']==2+len(policy['stages'])
            assert reuse['counts'].get('decision_reuse',0)==(180 if 'decision'in policy['stages']else 0)
            if label in('z1-both-r2','z1-world-r2'):assert row['psnr_passed']
    failed=evidence/'z1-both-r1-interrupted';summary=load(failed/'summary.json')
    assert not summary['clean']and summary['interrupted_due_to_TF32_stage_label_bug']and summary['process_wall_s']is None
    assert 'KeyboardInterrupt'in(failed/'guard.log').read_text()
    public=load(evidence/'public-repro-cpu-r1/summary.json');assert public['passed']and public['no_GPU_execution']
    assert len(public['trials'])==7 and all(r['exact_timed_manifest_and_assets']for r in public['trials'])
    assert public['public_script_SHA256']==sha(repo/'reproduce/run_temporal_live_trial.py')
    print('Temporal live trials: actual49 adjacent pairs/timings/unchanged23 videos;live12-owner quality failure;full Z1 reuse evidence and public source reconstruction verified.')


if __name__=='__main__':verify()
