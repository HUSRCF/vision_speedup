"""Verify exact predictor sources, complete workloads and independent quality."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    code = repo/'experiments/temporal-prediction61'
    evidence = repo/'results/aggressive-20261029/temporal-prediction61'
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    expected_labels = {'deep13-r1', 'residual23-r1'}
    assert {p.name for p in code.iterdir()} == expected_labels
    for label in sorted(expected_labels):
        frozen = code/label
        metadata = load(frozen/'trial.json')
        overlay = frozen/'production-overlay'
        base = repo/metadata['base']
        manifest = load(overlay/'crossstep211-production-sources.json')
        original = load(base/'nonv2-aggressive-production-sources.json')
        assert metadata['source_count'] == len(manifest) == 61
        assert {n for n,h in original.items() if manifest[n] != h} == {'src/unifolm_wma/utils/cuda_graph.py'}
        assert set(manifest)-set(original) == {
            'src/unifolm_wma/utils/crossstep_feature_runtime.py',
            'src/unifolm_wma/utils/crossstep_feature_reuse.py'}
        for name,digest in manifest.items():
            p = overlay/name
            assert sha(p if p.exists() else base/name) == digest, name
        for name,digest in load(frozen/'asset-hashes.json').items():
            assert sha(overlay/name) == digest, name
        for name,digest in load(frozen/'guard-source-hashes.json').items():
            assert sha(frozen/'guard'/name) == digest, name
        assert load(frozen/'guard/flags.json') == load(base.parent/(metadata['case']+'-flags.json'))
        assert (metadata['rounds'], metadata['videos']) == (12, 25)
        policy = metadata['feature_policy']
        assert policy['stages'] == ['decision', 'world']
        assert policy['start'] == 10 and policy['stop'] == 40
        assert policy['scope'] == 'deep' and policy['boundary_residual'] == (label == 'residual23-r1')
        prediction = metadata['prediction_policy']
        assert prediction['history'] == 2 and prediction['actual_position_gap_ratio']
        assert prediction['first_reuse_alpha'] == 1. and prediction['subsequent_reuse_alpha'] == .5
        assert prediction['all_rounds_and50_steps_preserved'] and prediction['model_quality_not_inferred_from_CPU']
        source = (overlay/'src/unifolm_wma/utils/crossstep_feature_reuse.py').read_text()
        assert 'EXPERIMENT_TAG=1793264401' in source and 'enable_fp_fusion=False' in source
        assert 'alpha=(position-last)/(last-before)' in source
        assert 'value.float()-self.anchor.float()' in source
        cpu = (frozen/'guard/cpu-preflight.log').read_text()
        assert 'affine_complete_outputs=3600' in cpu and 'two_fresh_after_invalidation=true' in cpu
        assert 'NONV2_EXPLICIT_STAGE_CPU passed=true independent_of_TF32=true' in cpu
        full = evidence/label
        summary = load(full/'summary.json')
        assert not summary['official_adopted'] and summary['verified_source_count'] == 61
        assert summary['clean'] and summary['workload_complete']
        guard = load(full/'guard-report.json')
        assert guard['clean']
        report = guard['diagnostic']
        row = next(r for r in load(full/'run/result.json')['cases'] if r['case'] == metadata['case'])
        assert row == report['case_result']
        assert row['rounds'] == 12 and row['videos'] == 25 and row['execution_ok'] and row['profile_complete']
        assert row['process_wall_s'] == summary['process_wall_s'] and row['psnr_db'] == summary['psnr_db']
        assert row['psnr_passed'] == summary['quality_passed'] == (row['psnr_db'] > 25)
        profile = load(full/'run/profiles'/f"{metadata['case']}.json")
        assert profile['total_process_wall_s'] == row['process_wall_s']
        reuse = report['crossstep']
        assert reuse['predictive_feature_reuse'] and reuse['history_complete_refreshes'] == 2
        assert reuse['graph_count'] == 4
        assert reuse['alpha_histogram'] == {'1.0':24, '0.5':336}
        assert len(reuse['selected_blocks']) == (13 if label == 'deep13-r1' else 23)
        assert reuse['boundary_residual'] == reuse['current_boundary_input_live'] == policy['boundary_residual']
        assert reuse['counts'] == dict(decision_calls=600, decision_refresh=420, decision_reuse=180,
                                        world_calls=600, world_refresh=420, world_reuse=180)
        assert len(reuse['completed_generations']) == 24
        assert all(g['steps'] == 50 for g in reuse['completed_generations'])
        assert not reuse['round_count_reduced'] and not reuse['sampler_step_count_reduced']
        check = reuse['graph_prediction_selfcheck']
        assert check['passed'] and len(check['cases']) == 4
        assert {(r['dtype'],r['residual']) for r in check['cases']} == {
            (dtype,residual) for dtype in ('torch.float16', 'torch.float32') for residual in (False, True)}
        assert all(r['tail'] == 257 and r['mutated_replay_bitwise'] for r in check['cases'])
    public = load(evidence/'public-repro-cpu-r1/summary.json')
    assert public['passed'] and public['no_GPU_execution']
    assert {r['trial'] for r in public['trials']} == expected_labels
    assert all(r['source_count'] == 61 and r['exact_timed_manifest_and_assets']
                and r['predictor_affine3600_CPU_passed'] and r['stage_independent_of_TF32'] for r in public['trials'])
    assert public['public_script_SHA256'] == sha(repo/'reproduce/run_temporal_prediction61.py')
    print('Two-refresh prediction61: two exact public rebuilds,affine CPU and mutated CUDA graphs,full12/25 workloads and independent quality results verified.')


if __name__ == '__main__':
    verify()
