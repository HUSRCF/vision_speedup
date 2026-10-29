"""Verify fresh dual-arm tail sharding, original control and accepted103 bundle."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    family = repo/'experiments/nonv2-other-tail66'
    evidence = repo/'results/aggressive-20261029/nonv2-other-tail66'
    hashes = []
    timings = []
    for label, parent, count, seconds in (
        ('dual-full-r3', family/'dual-full-r1', 66, 84.195081),
        ('dual-original65-control', repo/'experiments/nonv2-late23-prediction65/dual-full-r1', 65, 88.227722),
    ):
        identity = load(family/label/'source-identity.json')
        assert identity['manifest'] == load(parent/'production-sources.json')
        assert identity['asset_hashes'] == load(parent/'timing-asset-hashes.json')
        assert identity['source_count'] == count
        assert identity['exact_original_source_and_assets'] and identity['fresh_output_and_guard_state']
        if count == 66:
            assert identity['script_hashes'] == load(parent/'guard-source-hashes.json')
            for name, digest in identity['script_hashes'].items():
                assert sha(family/label/'preflight'/name) == digest
        run = evidence/label
        summary = load(run/'summary.json')
        assert all(summary[k] for k in ('clean', 'quality_passed', 'workload_complete', 'rank_videos_exact'))
        assert summary['process_wall_s'] == seconds and summary['psnr_db'] == 26.556920797050235
        assert summary['gpu_candidates'] == ['1', '3'] and summary['verified_source_count'] == count
        row = next(r for r in load(run/'result.json')['cases'] if r['case'] == 'unitree_z1_dual_arm_stackbox')
        assert row['rounds'] == 7 and row['videos'] == 15 and row['execution_ok'] and row['profile_complete']
        assert row['psnr_passed'] and row['process_wall_s'] == seconds
        complete = load(run/'parallel/parallel-complete.json')
        assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal']
        assert len(complete['video_hashes']) == 15 and complete['stage_calls'] == dict(decision=350, world=350)
        hashes.append(complete['video_hashes'])
        timings.append(seconds)
        for rank in (0, 1):
            ctl = load(run/f'parallel/rank{rank}-controller.json')
            assert ctl['stage_captures'] == dict(decision=36, world=36) and ctl['controller']['captured'] == 72
            assert {k: len(v) for k, v in ctl['graph_role_modules'].items()} == dict(
                decision_full=33, decision_reuse=3, world_full=33, world_reuse=3)
            if count == 66:
                assert ctl['controller']['resblock_frames']['captured'] == 12
                assert {k: len(v) for k, v in ctl['resblock_graph_role_modules'].items()} == dict(
                    decision_full=5, decision_reuse=1, world_full=5, world_reuse=1)
            feature = ctl['feature_reuse']
            assert feature['window'] == [20, 40] and len(feature['selected_blocks']) == 23
            assert feature['counts'] == dict(decision_calls=350, decision_refresh=280, decision_reuse=70,
                world_calls=350, world_refresh=280, world_reuse=70)
            assert feature['alpha_histogram'] == {'1.0': 14, '0.5': 126}
            assert len(feature['completed_generations']) == 14 and all(g['steps'] == 50 for g in feature['completed_generations'])
            assert not feature['round_count_reduced'] and not feature['sampler_step_count_reduced']
    original = load(repo/'results/aggressive-20261029/nonv2-late23-prediction65/dual-full-r1/parallel/parallel-complete.json')['video_hashes']
    assert hashes[0] == hashes[1] == original
    comparison = load(evidence/'dual-r3-complete-process-comparison.json')
    assert comparison['one_complete_candidate_control_pair'] and not comparison['multiple_independent_matched_pairs']
    assert comparison['candidate_s'] == timings[0] and comparison['baseline_s'] == timings[1]
    assert comparison['saved_s'] == timings[1]-timings[0] > 0
    adopted = load(family/'dual-r3-accepted-outcome.json')
    official = evidence/'dual-r3-official103'
    history = repo/'results/history/official-before-dual-tail66'
    assert adopted['official_adopted'] and adopted['official103_accepted'] and adopted['local_GPU_pair_updated']
    assert not adopted['website_uploaded'] and adopted['source_count'] == 66 and adopted['GPUs'] == 2
    assert adopted['all103_video_SHA_unchanged'] and adopted['selected_case_videos_formal_SHA_exact']
    assert adopted['case_sum_s'] == 715.704516 and adopted['previous_case_sum_s'] == 716.086762
    assert adopted['formal_saved_s'] == .382246 and adopted['selected_video_count'] == 15 and adopted['unchanged_video_count'] == 88
    assert adopted['selected_cases'] == {'unitree_z1_dual_arm_stackbox': 'nonv2_other_tail66_dual_full_r3'}
    for name, digest in adopted['pair_hashes'].items():
        assert sha(repo/'results'/name) == sha(official/'validated'/name) == digest
    for name, digest in adopted['previous_pair_hashes'].items():
        assert sha(history/name) == digest
    for name, digest in adopted['assembly_script_hashes'].items():
        assert sha(official/name) == digest
    fresh = load(official/'validated/result.json')
    previous = load(history/'result.json')
    assert fresh == load(repo/'results/result.json') and fresh['status'] == 'accepted'
    assert fresh['psnr_validation']['video_count'] == 103 and sum(r['rounds'] for r in fresh['cases']) == 49
    for row in fresh['cases']:
        old = next(r for r in previous['cases'] if r['case'] == row['case'])
        if row['case'] == 'unitree_z1_dual_arm_stackbox':
            assert row['process_wall_s'] == timings[0] and row['psnr_db'] == old['psnr_db']
        else:
            assert row == old
    normalize = lambda value: {p.rsplit('/outputs/', 1)[1]: h for p, h in value['hashes'].items()}
    links = load(official/'validated/immutable-video-links.json')
    prior = load(repo/'results/aggressive-20261029/nonv2-stop46/official103/validated/immutable-video-links.json')
    assert len(normalize(links)) == 103 and normalize(links) == normalize(prior)
    assert links['ordinary_files'] and links['original_outputs_unchanged'] and links['no_GPU_rerun_or_timer_rewrite']
    print('DUAL_TAIL66_R3_ADOPTED fresh_full_control=true all15_SHA_exact=true fresh103_accepted=true all103_SHA_unchanged=true')


if __name__ == '__main__':
    verify()
