"""Check frozen source identity and CPU evidence, then any completed GPU trial."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda path: json.loads(path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    frozen = repo/'experiments/nonv2-pencils-window65/full-r1'
    overlay = frozen/'production-overlay'
    previous = repo/'experiments/nonv2-other-prediction65/pencils-full-r1'
    original = load(previous/'production-sources.json')
    manifest = load(frozen/'production-sources.json')
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    assert len(manifest) == 65 and set(manifest) == set(original)
    assert {name for name in manifest if manifest[name] != original[name]} == {runtime}
    for name, digest in manifest.items():
        path = overlay/name
        assert sha(path if path.exists() else base/name) == digest, name
    for name, digest in load(frozen/'timing-asset-hashes.json').items():
        assert sha(overlay/name) == digest, name
    for name, digest in load(frozen/'guard-source-hashes.json').items():
        assert sha(frozen/name) == digest, name
    assert load(frozen/'flags.json') == load(previous/'flags.json')
    source = (overlay/runtime).read_text()
    assert source == (previous/'production-overlay'/runtime).read_text().replace(
        '==15*args.n_iter', '==10*args.n_iter')
    entry_name = 'scripts/evaluation/world_model_interaction.py'
    entry = (overlay/entry_name).read_text()
    assert entry == (previous/'production-overlay'/entry_name).read_text().replace(
        "'start': 10, 'stop': 40", "'start': 20, 'stop': 40")
    trial = load(frozen/'trial.json')
    assert (trial['rounds'], trial['videos'], trial['GPUs']) == (8, 17, 2)
    policy = trial['feature_policy']
    assert (policy['start'], policy['stop']) == (20, 40)
    assert policy['stages'] == ['decision', 'world']
    assert not trial['official_adopted']
    assert trial['per_role_expected_gathers'] == dict(
        decision_full=33, decision_reuse=21, world_full=33, world_reuse=21)
    proof = load(frozen/'timing-proof.json')
    assert proof['prediction_steps_per_stage_per_round'] == 10
    assert proof['original50_steps_and_rounds_preserved']
    evidence = repo/'results/aggressive-20261029/nonv2-pencils-window65'
    cpu = load(evidence/'public-repro-cpu-r1/summary.json')
    for field in ('passed', 'no_GPU_execution', 'exact_timed_manifest_and_assets',
                  'predictor_affine3600_CPU_passed', 'lexical_cache_scope_passed',
                  'all1200_calls_through_parallel_executor', 'window20_40_CPU_checked'):
        assert cpu[field], field
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_nonv2_pencils_window65.py')
    summary = evidence/'full-r1/summary.json'
    if summary.exists():
        result = load(summary)
        if result['clean']:
            assert result['workload_complete'] and result['rank_videos_exact']
            run = summary.parent
            row = next(row for row in load(run/'result.json')['cases']
                       if row['case'] == trial['case'])
            assert (row['rounds'], row['videos']) == (8, 17)
            assert row['process_wall_s'] == result['process_wall_s']
            assert row['psnr_db'] == result['psnr_db']
            assert row['psnr_passed'] == result['quality_passed']
            for rank in (0, 1):
                controller = load(run/f'parallel/rank{rank}-controller.json')
                assert controller['stage_captures'] == dict(decision=54, world=54)
                assert controller['controller']['captured'] == 108
                feature = controller['feature_reuse']
                assert feature['window'] == [20, 40]
                assert feature['graph_count'] == 4
                assert feature['alpha_histogram'] == {'1.0': 16, '0.5': 144}
                assert feature['counts']['decision_calls'] == feature['counts']['world_calls'] == 400
                assert feature['counts']['decision_reuse'] == feature['counts']['world_reuse'] == 80
                assert feature['graph_prediction_selfcheck']['passed']
                assert not feature['sampler_step_count_reduced'] and not feature['round_count_reduced']
    print('PENCILS_WINDOW65_VERIFIED source65=true original_predictor_and_math=true CPU_only_quality_not_inferred=true')


if __name__ == '__main__':
    verify()
