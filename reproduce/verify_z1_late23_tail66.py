"""Validate the deeper Z1 reuse experiment without inferring video quality from CPU tests."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    frozen = repo/'experiments/z1-late23-tail66/full-r1'
    evidence = repo/'results/aggressive-20261029/z1-late23-tail66'
    parent = repo/'experiments/nonv2-other-tail66/z1-full-r1'
    overlay = frozen/'production-overlay'
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    manifest = load(frozen/'production-sources.json')
    old = load(parent/'production-sources.json')
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    entry = 'scripts/evaluation/world_model_interaction.py'
    assert set(manifest) == set(old) and len(manifest) == 66
    assert {n for n, h in old.items() if manifest[n] != h} == {runtime}
    for name, h in manifest.items():
        p = overlay/name
        assert sha(p if p.exists() else base/name) == h, name
    assets = load(frozen/'timing-asset-hashes.json')
    old_assets = load(parent/'timing-asset-hashes.json')
    assert set(assets) == set(old_assets)
    assert {n for n, h in old_assets.items() if assets[n] != h} == {entry, 'nccl-entry-timing-proof.json'}
    for name, h in assets.items(): assert sha(overlay/name) == h
    for name, h in load(frozen/'guard-source-hashes.json').items(): assert sha(frozen/name) == h
    before = (parent/'production-overlay'/entry).read_text()
    after = (overlay/entry).read_text()
    old_policy = "{'input_start': 6, 'output_end': 6, 'start': 10, 'stop': 40, 'boundary_residual': False, 'scope': 'deep', 'stages': ('decision', 'world')}"
    new_policy = "{'input_start': 1, 'output_end': 11, 'start': 20, 'stop': 40, 'boundary_residual': False, 'scope': 'deep', 'stages': ('decision', 'world')}"
    assert before.count(old_policy) == 1 and after == before.replace(old_policy, new_policy)
    expected = (parent/'production-overlay'/runtime).read_text()
    for old_text, new_text in [('==15*args.n_iter', '==10*args.n_iter'),
                               ('decision_full=5,decision_reuse=5,world_full=5,world_reuse=5', 'decision_full=5,decision_reuse=1,world_full=5,world_reuse=1'),
                               ("report['resblock_frames']['captured']==20", "report['resblock_frames']['captured']==12")]:
        assert expected.count(old_text) == 1
        expected = expected.replace(old_text, new_text)
    assert (overlay/runtime).read_text() == expected
    cpu = load(evidence/'public-repro-cpu-r1/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution'] and len(cpu['trials']) == 1
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_z1_late23_tail66.py')
    for name, h in cpu['logs_SHA256'].items(): assert sha(evidence/'public-repro-cpu-r1'/name) == h
    text = (evidence/'public-repro-cpu-r1/z1-check_nonv2_crossstep65_cpu.py.log').read_text()
    assert 'Z1_LATE23_TAIL66_CPU passed=true actual23_parents=true all1200_affine_outputs_exact=true' in text
    assert 'PREDICT_CACHE_SCOPE_CPU passed=true' in text and 'affine_complete_outputs=3600' in text
    failed_cpu = load(evidence/'initial-cpu-preflight-failed/summary.json')
    assert failed_cpu['no_GPU_execution'] and failed_cpu['test_observation_error']
    assert failed_cpu['inference_code_not_changed_by_test_fix']
    first = load(evidence/'full-r1/summary.json')
    assert first['timing_excluded'] and not first['clean'] and first['process_wall_s'] is None
    assert first['queue_retirement_proof']['no_foreign_process_signaled']
    assert first['queue_retirement_proof']['attempt2_not_started']
    assert list((evidence/'full-r1').rglob('timing-contaminated.txt'))
    identity = load(repo/'experiments/z1-late23-tail66/full-r2/source-identity.json')
    assert identity['manifest'] == manifest and identity['asset_hashes'] == assets
    assert identity['all_source_and_assets_SHA_unchanged'] and identity['fresh_output_and_guard_state']
    run = evidence/'full-r2'; summary = load(run/'summary.json')
    assert summary['clean'] and summary['workload_complete'] and summary['rank_videos_exact']
    assert summary['gpu_candidates'] == ['0', '2'] and summary['verified_source_count'] == 66
    row = next(x for x in load(run/'result.json')['cases'] if x['case'] == 'unitree_z1_stackbox')
    assert row['rounds'] == 12 and row['videos'] == 25 and row['execution_ok'] and row['profile_complete']
    assert row['process_wall_s'] == summary['process_wall_s'] and row['psnr_db'] == summary['psnr_db']
    assert row['psnr_passed'] == summary['quality_passed'] == (row['psnr_db'] > 25)
    complete = load(run/'parallel/parallel-complete.json')
    assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal'] and len(complete['video_hashes']) == 25
    assert complete['stage_calls'] == dict(decision=600, world=600)
    for rank in (0, 1):
        ctl = load(run/f'parallel/rank{rank}-controller.json')
        assert ctl['stage_captures'] == dict(decision=36, world=36) and ctl['controller']['captured'] == 72
        assert ctl['controller']['resblock_frames']['captured'] == 12
        assert {k: len(v) for k, v in ctl['graph_role_modules'].items()} == dict(decision_full=33, decision_reuse=3, world_full=33, world_reuse=3)
        assert {k: len(v) for k, v in ctl['resblock_graph_role_modules'].items()} == dict(decision_full=5, decision_reuse=1, world_full=5, world_reuse=1)
        feature = ctl['feature_reuse']
        assert len(feature['selected_blocks']) == 23 and feature['graph_count'] == 4
        assert feature['graph_prediction_selfcheck']['passed']
        assert feature['counts'] == dict(decision_calls=600, decision_refresh=480, decision_reuse=120,
                                         world_calls=600, world_refresh=480, world_reuse=120)
        assert feature['alpha_histogram'] == {'1.0': 24, '0.5': 216}
        assert not feature['round_count_reduced'] and not feature['sampler_step_count_reduced']
    baseline = evidence/'baseline-full-r1'
    control = load(baseline/'summary.json')
    control_identity = load(repo/'experiments/z1-late23-tail66/baseline-full-r1/source-identity.json')
    assert control_identity['manifest'] == old and control_identity['asset_hashes'] == old_assets
    assert control['clean'] and control['quality_passed'] and control['workload_complete']
    assert control['rank_videos_exact'] and control['gpu_candidates'] == ['0', '2']
    assert control['verified_source_count'] == 66 and not control['official_adopted']
    comparison = load(evidence/'complete-process-comparison.json')
    assert comparison['one_complete_candidate_control_pair'] and not comparison['multiple_independent_matched_pairs']
    assert comparison['GPUs'] == ['0', '2'] and comparison['candidate_then_baseline']
    assert comparison['candidate_s'] == summary['process_wall_s'] == 132.539858
    assert comparison['baseline_s'] == control['process_wall_s']
    assert comparison['saved_s'] == control['process_wall_s']-summary['process_wall_s'] > 0
    control_complete = load(baseline/'parallel/parallel-complete.json')
    old_complete = load(repo/'results/aggressive-20261029/nonv2-other-tail66/z1-full-r2/parallel/parallel-complete.json')
    assert control_complete['video_hashes'] == old_complete['video_hashes']
    adopted = load(repo/'experiments/z1-late23-tail66/accepted-outcome.json')
    official = evidence/'official103'; history = repo/'results/history/official-before-z1-late23-tail66'
    assert adopted['official_adopted'] and adopted['official103_accepted'] and adopted['local_GPU_pair_updated']
    assert adopted['source_count'] == 66 and adopted['GPUs'] == 2 and not adopted['website_uploaded']
    assert adopted['all_rounds_and50_steps_preserved'] and adopted['unchanged_other78_video_SHA']
    assert adopted['selected25_new_actual_timed_videos'] and adopted['changed_video_count'] > 0
    assert adopted['selected_video_count'] == 25 and adopted['unchanged_video_count'] == 78
    assert adopted['case_sum_s'] == 722.609176 and adopted['previous_case_sum_s'] == 729.014469
    assert adopted['formal_saved_s'] == 6.405293
    for name, h in adopted['assembly_script_hashes'].items(): assert sha(official/name) == h
    for name, h in adopted['pair_hashes'].items(): assert sha(repo/'results/history/official-before-z1-reuse-window66'/name) == sha(official/'validated'/name) == h
    for name, h in adopted['previous_pair_hashes'].items(): assert sha(history/name) == h
    fresh = load(official/'validated/result.json')
    assert fresh == load(repo/'results/history/official-before-z1-reuse-window66/result.json') and fresh['status'] == 'accepted'
    assert fresh['psnr_validation']['video_count'] == 103 and sum(x['rounds'] for x in fresh['cases']) == 49
    previous = load(history/'result.json')
    for item in fresh['cases']:
        before = next(x for x in previous['cases'] if x['case'] == item['case'])
        if item['case'] == 'unitree_z1_stackbox':
            assert item['process_wall_s'] == 132.539858 and item['psnr_db'] == 30.332499792023174
            assert item['psnr_passed'] and item['process_wall_s'] < before['process_wall_s']
        else: assert item == before
    links = load(official/'validated/immutable-video-links.json')
    prior = load(evidence.parent/'nonv2-other-tail66/official103/validated/immutable-video-links.json')
    normalize = lambda row: {p.rsplit('/outputs/', 1)[1]: h for p, h in row['hashes'].items()}
    current, old_links = normalize(links), normalize(prior)
    assert len(current) == len(old_links) == 103 and current.keys() == old_links.keys()
    others = {n: h for n, h in current.items() if not n.startswith('unitree_z1_stackbox/')}
    assert len(others) == 78 and all(old_links[n] == h for n, h in others.items())
    assert sum(old_links[n] != h for n, h in current.items()) == adopted['changed_video_count']
    assert links['ordinary_files'] and links['original_outputs_unchanged'] and links['no_GPU_rerun_or_timer_rewrite']
    print('Z1_LATE23_TAIL66_VERIFIED source66=true unchanged_predictor_math=true full12_rounds_50steps=true PSNR=' + str(summary['psnr_db']))


if __name__ == '__main__': verify()
