"""Check per-task tail sharding, unchanged predictors, and full-run evidence."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    family = repo/'experiments/nonv2-other-tail66'
    evidence = repo/'results/aggressive-20261029/nonv2-other-tail66'
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    helper = 'src/unifolm_wma/utils/nonv2_resblock_tail73_mode.py'
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    pencils = repo/'experiments/nonv2-resblock-tail73/full66-r1/production-overlay'
    cpu = load(evidence/'public-repro-cpu-r1/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution']
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_nonv2_other_tail66.py')
    assert {x['case_alias'] for x in cpu['trials']} == {'z1', 'dual'}
    for row in cpu['trials']:
        assert all(row[k] for k in ('exact_source66_and_assets', 'original_executor_and_affine3600_CPU_checked',
                                   'lexical_cache_scope_restored', 'frame_and_temporal_boundary_CPU_checked', 'no_GPU_execution'))
    for name, h in cpu['logs_SHA256'].items(): assert sha(evidence/'public-repro-cpu-r1'/name) == h
    configs = {
        'z1': ('nonv2-prediction65/full-r2', 12, 15, 21, 5, 138.945151, 28.992932876074295),
        'dual': ('nonv2-late23-prediction65/dual-full-r1', 7, 10, 3, 1, 83.697401, 26.556920797050235),
    }
    for alias, (parent_name, rounds, predictions, retained, extra, seconds, psnr) in configs.items():
        frozen = family/(alias+'-full-r1'); overlay = frozen/'production-overlay'
        parent = repo/'experiments'/parent_name
        old = load(parent/'production-sources.json'); manifest = load(frozen/'production-sources.json')
        assert len(old) == 65 and len(manifest) == 66
        assert set(manifest)-set(old) == {helper}
        assert {n for n, h in old.items() if manifest[n] != h} == {runtime}
        assert sha(overlay/helper) == sha(pencils/helper)
        assets = load(frozen/'timing-asset-hashes.json')
        assert assets['scripts/evaluation/world_model_interaction.py'] == load(parent/'timing-asset-hashes.json')['scripts/evaluation/world_model_interaction.py']
        for name, h in manifest.items():
            path = overlay/name
            assert sha(path if path.exists() else base/name) == h, name
        for name, h in assets.items(): assert sha(overlay/name) == h
        for name, h in load(frozen/'guard-source-hashes.json').items(): assert sha(frozen/name) == h
        trial = load(frozen/'trial.json')
        assert trial['rounds'] == rounds and trial['videos'] == 2*rounds+1
        assert trial['pencils_quality_not_inherited'] and trial['original_sampler_and_predictor_unchanged']
        assert trial['per_role_expected_gathers'] == dict(decision_full=38, decision_reuse=retained+extra,
                                                        world_full=38, world_reuse=retained+extra)
        identity = load(family/(alias+'-full-r2/source-identity.json'))
        assert identity['manifest'] == manifest and identity['asset_hashes'] == assets
        assert identity['source_count'] == 66 and identity['fresh_output_and_guard_state']
        run = evidence/(alias+'-full-r2'); summary = load(run/'summary.json')
        assert all(summary[k] for k in ('clean', 'quality_passed', 'workload_complete', 'rank_videos_exact', 'all_source_and_assets_SHA_unchanged'))
        assert summary['process_wall_s'] == seconds and summary['psnr_db'] == psnr
        assert summary['gpu_candidates'] == ['1', '3'] and summary['verified_source_count'] == 66
        row = next(x for x in load(run/'result.json')['cases'] if x['case'] == trial['case'])
        assert row['rounds'] == rounds and row['videos'] == 2*rounds+1
        assert row['psnr_passed'] and row['profile_complete'] and row['execution_ok']
        complete = load(run/'parallel/parallel-complete.json')
        assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal']
        assert len(complete['video_hashes']) == 2*rounds+1
        assert complete['stage_calls'] == dict(decision=50*rounds, world=50*rounds)
        old_run = repo/'results/aggressive-20261029'/parent_name
        assert complete['video_hashes'] == load(old_run/'parallel/parallel-complete.json')['video_hashes']
        for rank in (0, 1):
            ctl = load(run/f'parallel/rank{rank}-controller.json')
            assert ctl['stage_captures'] == dict(decision=33+retained, world=33+retained)
            assert ctl['controller']['captured'] == 2*(33+retained)
            assert ctl['controller']['resblock_frames']['captured'] == 2*(5+extra)
            assert {role: len(names) for role, names in ctl['resblock_graph_role_modules'].items()} == dict(decision_full=5, decision_reuse=extra, world_full=5, world_reuse=extra)
            assert {role: len(names) for role, names in ctl['graph_role_modules'].items()} == dict(decision_full=33, decision_reuse=retained, world_full=33, world_reuse=retained)
            feature = ctl['feature_reuse']; calls = 50*rounds; reuses = predictions*rounds
            assert feature['graph_count'] == 4 and feature['graph_prediction_selfcheck']['passed']
            assert feature['counts'] == dict(decision_calls=calls, decision_refresh=calls-reuses, decision_reuse=reuses,
                                             world_calls=calls, world_refresh=calls-reuses, world_reuse=reuses)
            assert not feature['round_count_reduced'] and not feature['sampler_step_count_reduced']
    excluded = load(evidence/'z1-full-r1/summary.json')
    assert not excluded['clean'] and excluded['timing_excluded'] and excluded['process_wall_s'] is None
    assert excluded['pair_migration']['no_foreign_process_signaled']
    assert list((evidence/'z1-full-r1').rglob('timing-contaminated.txt'))
    initial_dual = load(evidence/'dual-full-r1/summary.json')
    assert initial_dual['clean'] and initial_dual['quality_passed']
    assert initial_dual['process_wall_s'] == 86.092845 and not initial_dual['official_adopted']
    comparison = load(evidence/'complete-process-comparison.json')
    assert set(comparison['cases']) == {'z1'} and comparison['dual_unadopted']
    assert comparison['dual_two_clean_candidate_times_s'] == [86.092845, 83.697401]
    assert comparison['dual_control_contaminated_excluded'] and not comparison['multiple_independent_matched_pairs']
    failed_control = load(evidence/'dual-baseline-full-r1/summary.json')
    assert failed_control['timing_excluded'] and not failed_control['clean']
    assert failed_control['process_wall_s'] is None
    assert failed_control['control_retirement']['no_foreign_process_signaled']
    assert failed_control['control_retirement']['attempt2_not_started']
    assert list((evidence/'dual-baseline-full-r1').rglob('timing-contaminated.txt'))
    for alias, row in comparison['cases'].items():
        assert row['GPUs'] == ['1', '3'] and row['one_complete_candidate_control_pair']
        candidate = load(evidence/(alias+'-full-r2/summary.json'))
        baseline = load(evidence/(alias+'-baseline-full-r1/summary.json'))
        assert baseline['clean'] and baseline['quality_passed'] and baseline['verified_source_count'] == 65
        assert row['candidate_s'] == candidate['process_wall_s'] and row['baseline_s'] == baseline['process_wall_s']
        assert row['saved_s'] == row['baseline_s']-row['candidate_s'] > 0
        assert load(evidence/(alias+'-baseline-full-r1/parallel/parallel-complete.json'))['video_hashes'] == load(evidence/(alias+'-full-r2/parallel/parallel-complete.json'))['video_hashes']
        parent = repo/'experiments'/configs[alias][0]
        identity = load(family/(alias+'-baseline-full-r1/source-identity.json'))
        assert identity['manifest'] == load(parent/'production-sources.json')
        assert identity['asset_hashes'] == load(parent/'timing-asset-hashes.json')
    adopted = load(family/'accepted-outcome.json'); official = evidence/'official103'
    assert adopted['official_adopted'] and adopted['official103_accepted'] and adopted['local_GPU_pair_updated']
    assert adopted['source_count'] == 66 and adopted['GPUs'] == 2 and not adopted['website_uploaded']
    assert adopted['all_rounds_and50_steps_preserved'] and adopted['all103_video_SHA_unchanged']
    assert adopted['selected_case_videos_formal_SHA_exact']
    assert set(adopted['selected_cases']) == {'unitree_z1_stackbox'}
    assert adopted['selected_video_count'] == 25 and adopted['unchanged_video_count'] == 78
    for name, h in adopted['assembly_script_hashes'].items(): assert sha(official/name) == h
    fresh = load(official/'validated/result.json')
    assert fresh == load(repo/'results/history/official-before-z1-late23-tail66/result.json') and fresh['status'] == 'accepted'
    assert fresh['psnr_validation']['video_count'] == 103
    history = repo/'results/history/official-before-other-tail66'
    for name, h in adopted['pair_hashes'].items(): assert sha(repo/'results/history/official-before-z1-late23-tail66'/name) == sha(official/'validated'/name) == h
    for name, h in adopted['previous_pair_hashes'].items(): assert sha(history/name) == h
    previous = load(history/'result.json')
    assert adopted['previous_case_sum_s'] == 735.397241
    assert abs(sum(x['process_wall_s'] for x in fresh['cases'])-adopted['case_sum_s']) < 1e-6
    assert adopted['formal_saved_s'] == round(735.397241-adopted['case_sum_s'], 6) > 0
    for row in fresh['cases']:
        before = next(x for x in previous['cases'] if x['case'] == row['case'])
        if row['case'] in adopted['cases']:
            delta = adopted['cases'][row['case']]
            assert row['process_wall_s'] == delta['current_s'] < before['process_wall_s']
            assert row['psnr_db'] == delta['psnr_db'] == before['psnr_db']
        else: assert row == before
    links = load(official/'validated/immutable-video-links.json')
    prior = load(repo/'results/aggressive-20261029/nonv2-resblock-tail73/official103/validated/immutable-video-links.json')
    normalize = lambda x: {p.rsplit('/outputs/', 1)[1]: h for p, h in x['hashes'].items()}
    assert len(normalize(links)) == 103 and normalize(links) == normalize(prior)
    assert links['ordinary_files'] and links['original_outputs_unchanged'] and links['no_GPU_rerun_or_timer_rewrite']
    print('OTHER_TAIL66_VERIFIED predictors_unchanged=true full_rounds_50steps=true all103_SHA_unchanged=true fresh103_accepted=true')


if __name__ == '__main__': verify()
