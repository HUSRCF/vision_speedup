"""Check frozen source identity and CPU evidence, then any completed GPU trial."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda path: json.loads(path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    frozen = repo/'experiments/nonv2-late23-prediction65/full-r1'
    overlay = frozen/'production-overlay'
    previous = repo/'experiments/nonv2-pencils-window65/full-r1'
    original = load(previous/'production-sources.json')
    manifest = load(frozen/'production-sources.json')
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    assert len(manifest) == 65 and set(manifest) == set(original)
    assert manifest == original
    for name, digest in manifest.items():
        path = overlay/name
        assert sha(path if path.exists() else base/name) == digest, name
    for name, digest in load(frozen/'timing-asset-hashes.json').items():
        assert sha(overlay/name) == digest, name
    for name, digest in load(frozen/'guard-source-hashes.json').items():
        assert sha(frozen/name) == digest, name
    assert load(frozen/'flags.json') == load(previous/'flags.json')
    source = (overlay/runtime).read_text()
    assert source == (previous/'production-overlay'/runtime).read_text()
    entry_name = 'scripts/evaluation/world_model_interaction.py'
    entry = (overlay/entry_name).read_text()
    assert entry == (previous/'production-overlay'/entry_name).read_text().replace(
        "'input_start': 6, 'output_end': 6", "'input_start': 1, 'output_end': 11")
    trial = load(frozen/'trial.json')
    assert (trial['rounds'], trial['videos'], trial['GPUs']) == (8, 17, 2)
    policy = trial['feature_policy']
    assert (policy['start'], policy['stop']) == (20, 40)
    assert policy['stages'] == ['decision', 'world']
    assert not trial['official_adopted']
    assert trial['per_role_expected_gathers'] == dict(
        decision_full=33, decision_reuse=3, world_full=33, world_reuse=3)
    proof = load(frozen/'timing-proof.json')
    assert proof['prediction_steps_per_stage_per_round'] == 10
    assert proof['original50_steps_and_rounds_preserved']
    evidence = repo/'results/aggressive-20261029/nonv2-late23-prediction65'
    cpu = load(evidence/'public-repro-cpu-r1/summary.json')
    for field in ('passed', 'no_GPU_execution', 'exact_timed_manifest_and_assets',
                  'predictor_affine3600_CPU_passed', 'lexical_cache_scope_passed',
                  'all1200_calls_through_parallel_executor', 'window20_40_CPU_checked'):
        assert cpu[field], field
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_nonv2_late23_prediction65.py')
    assert cpu['selected_parent_blocks'] == 23 and cpu['per_reuse_step_gathers'] == 3
    summary = evidence/'full-r1/summary.json'
    assert summary.exists()
    if summary.exists():
        result = load(summary)
        if result['clean']:
            assert result['workload_complete'] and result['rank_videos_exact'] and result['quality_passed']
            assert result['process_wall_s'] == 95.640695 and result['psnr_db'] == 26.79860623954979
            complete = load(summary.parent/'parallel/parallel-complete.json')
            assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal']
            assert len(complete['video_hashes']) == 17
            assert complete['stage_calls'] == dict(decision=400, world=400)
            run = summary.parent
            row = next(row for row in load(run/'result.json')['cases']
                       if row['case'] == trial['case'])
            assert (row['rounds'], row['videos']) == (8, 17)
            assert row['process_wall_s'] == result['process_wall_s']
            assert row['psnr_db'] == result['psnr_db']
            assert row['psnr_passed'] == result['quality_passed']
            for rank in (0, 1):
                controller = load(run/f'parallel/rank{rank}-controller.json')
                assert controller['stage_captures'] == dict(decision=36, world=36)
                assert controller['controller']['captured'] == 72
                feature = controller['feature_reuse']
                assert feature['window'] == [20, 40]
                assert feature['graph_count'] == 4
                assert len(feature['selected_blocks']) == 23
                assert feature['history_complete_refreshes'] == 2
                assert {role:len(names) for role,names in controller['graph_role_modules'].items()} == trial['per_role_expected_gathers']
                assert feature['alpha_histogram'] == {'1.0': 16, '0.5': 144}
                assert feature['counts']['decision_calls'] == feature['counts']['world_calls'] == 400
                assert feature['counts']['decision_reuse'] == feature['counts']['world_reuse'] == 80
                assert feature['graph_prediction_selfcheck']['passed']
                assert not feature['sampler_step_count_reduced'] and not feature['round_count_reduced']
    accepted = load(repo/'experiments/nonv2-late23-prediction65/accepted-outcome.json')
    assert accepted['official_adopted'] and accepted['official103_accepted'] and accepted['local_GPU_pair_updated']
    assert accepted['source_count'] == 65 and accepted['GPUs'] == 2 and not accepted['website_uploaded']
    assert accepted['all_rounds_and50_steps_preserved'] and accepted['historical_comparison_not_matched_AB']
    assert accepted['case_sum_s'] == 744.785412 and accepted['previous_case_sum_s'] == 755.093960
    assert accepted['formal_saved_s'] == 10.308548
    assert set(accepted['selected_cases']) == {trial['case']}
    assert accepted['selected_video_count'] == 17 and accepted['unchanged_video_count'] == 86
    official = evidence/'official103'
    for name,digest in accepted['assembly_script_hashes'].items():
        assert sha(official/name) == digest
    fresh = load(official/'validated/result.json')
    assert fresh == load(repo/'results/history/official-before-dual-late23-prediction65/result.json') and fresh['status'] == 'accepted'
    assert fresh['psnr_validation']['video_count'] == 103
    history = repo/'results/history/official-before-late23-prediction65'
    for name,digest in accepted['pair_hashes'].items():
        assert sha(repo/'results/history/official-before-dual-late23-prediction65'/name) == sha(official/'validated'/name) == digest
    for name,digest in accepted['previous_pair_hashes'].items():
        assert sha(history/name) == digest
    before = load(history/'result.json')
    for row in fresh['cases']:
        old = next(x for x in before['cases'] if x['case'] == row['case'])
        if row['case'] == trial['case']:
            assert row['process_wall_s'] == result['process_wall_s'] and row['psnr_db'] == result['psnr_db']
        else:
            assert row == old
    links = load(official/'validated/immutable-video-links.json')
    prior = load(repo/'results/aggressive-20261029/nonv2-other-prediction65/official103/validated/immutable-video-links.json')
    others = lambda data: {p:h for p,h in data['hashes'].items() if '/'+trial['case']+'/' not in p}
    assert links['count'] == 103 and links['ordinary_files'] and links['original_outputs_unchanged']
    assert links['selected_actual_timed_videos'] == 17 and links['prior_accepted_SHA_equal_videos'] == 86
    assert links['no_GPU_rerun_or_timer_rewrite'] and len(others(links)) == 86 and others(links) == others(prior)
    print('LATE23_PREDICTION65_VERIFIED source65=true full17_quality_passed=true two_ranks_exact=true fresh103_accepted=true unchanged86=true')


if __name__ == '__main__':
    verify()
