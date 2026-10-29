"""Keep full quality failures separate from graph-startup failures and CPU checks."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    base = repo/'unifolm'
    original = load(base/'world-head-filter-cache-production-sources.json')
    evidence = repo/'results/aggressive-20261029/v2-prediction211'
    cpu = load(evidence/'public-repro-cpu-r2/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution']
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_v2_prediction211.py')
    assert {r['trial'] for r in cpu['trials']} == {'world-r1', 'both-r1', 'both-r2'}
    predictor = repo/'experiments/temporal-prediction61/deep13-r1/production-overlay/src/unifolm_wma/utils/crossstep_feature_reuse.py'
    for label in ('world-r1', 'both-r1', 'both-r2'):
        frozen = repo/'experiments/v2-prediction211'/label
        overlay = frozen/'production-overlay'
        manifest = load(overlay/'crossstep211-production-sources.json')
        assert len(manifest) == 211
        assert set(manifest)-set(original) == {
            'src/unifolm_wma/utils/crossstep_feature_reuse.py',
            'src/unifolm_wma/utils/crossstep_feature_runtime.py'}
        assert {n for n,h in original.items() if manifest[n] != h} == {
            'src/unifolm_wma/utils/cuda_graph.py'}
        for name,digest in manifest.items():
            p = overlay/name
            assert sha(p if p.exists() else base/name) == digest, name
        for name,digest in load(frozen/'asset-hashes.json').items():
            assert sha(overlay/name) == digest, name
        for name,digest in load(frozen/'guard-source-hashes.json').items():
            assert sha(frozen/'guard'/name) == digest, name
        assert sha(overlay/'src/unifolm_wma/utils/crossstep_feature_reuse.py') == sha(predictor)
        trial = load(frozen/'trial.json')
        assert (trial['rounds'], trial['videos'], trial['source_count']) == (11, 23, 211)
        assert trial['prediction_policy']['V2_full_refresh_math_and_FP32_attention_preserved']
        assert not trial['official_adopted']
        row = next(r for r in cpu['trials'] if r['trial'] == label)
        assert row['exact_timed_manifest_and_assets'] and row['predictor_affine3600_CPU_passed']
        assert row['stage_independent_of_TF32']
        summary = load(evidence/label/'summary.json')
        assert not summary['official_adopted']
        if label != 'world-r1':
            assert not summary['clean']
            assert summary['process_wall_s'] is None and summary['psnr_db'] is None
    run = evidence/'world-r1/run'
    summary = load(evidence/'world-r1/summary.json')
    assert summary['clean'] and summary['workload_complete'] and not summary['quality_passed']
    row = next(r for r in load(run/'result.json')['cases'] if r['case'].endswith('_v2'))
    assert (row['rounds'], row['videos']) == (11, 23)
    assert row['process_wall_s'] == summary['process_wall_s'] == 311.092018
    assert row['psnr_db'] == summary['psnr_db'] == 18.250677627674747
    assert not row['psnr_passed']
    feature = load(run/'profiles/unitree_z1_dual_arm_stackbox_v2.crossstep.json')
    assert feature['graph_count'] == 3 and feature['stages'] == ['world']
    assert feature['counts'] == dict(decision_calls=550, decision_refresh=550,
                                     world_calls=550, world_refresh=385, world_reuse=165)
    assert feature['alpha_histogram'] == {'1.0':11, '0.5':154}
    assert len(feature['completed_generations']) == 22
    assert all(r['steps'] == 50 for r in feature['completed_generations'])
    assert feature['graph_prediction_selfcheck']['passed']
    assert not feature['sampler_step_count_reduced'] and not feature['round_count_reduced']
    flags = load(run/'flags.json')
    assert flags['UNIFOLM_ATTENTION_DTYPE'] == 'fp32'
    print('V2_PREDICTION211_VERIFIED full_world_quality_failed=true both_startups_failed=true CPU_quality_not_inferred=true')


if __name__ == '__main__':
    verify()
