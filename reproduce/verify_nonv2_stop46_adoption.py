"""Verify the selected pencils window against a fresh complete original control."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    family = repo/'experiments/nonv2-stop46'
    evidence = repo/'results/aggressive-20261029/nonv2-stop46'
    parent = repo/'experiments/nonv2-resblock-tail73/full66-r1'
    selected = load(evidence/'pencils-full-r1/summary.json')
    baseline = evidence/'pencils-baseline-full-r1'
    control = load(baseline/'summary.json')
    for row in (selected, control):
        assert row['clean'] and row['quality_passed'] and row['workload_complete'] and row['rank_videos_exact']
        assert row['gpu_candidates'] == ['0', '1'] and row['verified_source_count'] == 66
    assert selected['process_wall_s'] == 90.806070 and selected['psnr_db'] == 25.79678602776073
    assert control['formal_video_SHA_exact'] and control['process_wall_s'] > selected['process_wall_s']
    identity = load(family/'pencils-baseline-full-r1/source-identity.json')
    assert identity['manifest'] == load(parent/'production-sources.json')
    assert identity['asset_hashes'] == load(parent/'timing-asset-hashes.json')
    assert identity['all_source_and_assets_SHA_unchanged'] and identity['fresh_output_and_guard_state']
    original = repo/'results/aggressive-20261029/nonv2-resblock-tail73/full66-r1'
    assert load(baseline/'parallel/parallel-complete.json')['video_hashes'] == load(original/'parallel/parallel-complete.json')['video_hashes']
    for rank in (0, 1):
        ctl = load(baseline/f'parallel/rank{rank}-controller.json')
        assert ctl['stage_captures'] == dict(decision=36, world=36)
        assert ctl['controller']['captured'] == 72
        feature = ctl['feature_reuse']
        assert feature['window'] == [20, 40] and len(feature['selected_blocks']) == 23
        assert feature['counts'] == dict(decision_calls=400, decision_refresh=320, decision_reuse=80,
                                        world_calls=400, world_refresh=320, world_reuse=80)
        assert feature['alpha_histogram'] == {'1.0': 16, '0.5': 144}
    comparison = load(evidence/'pencils-complete-process-comparison.json')
    assert comparison['one_complete_candidate_control_pair'] and not comparison['multiple_independent_matched_pairs']
    assert comparison['GPUs'] == ['0', '1'] and comparison['baseline_video_SHA_exact_previous_formal']
    assert comparison['candidate_s'] == selected['process_wall_s'] and comparison['baseline_s'] == control['process_wall_s']
    assert comparison['saved_s'] == control['process_wall_s']-selected['process_wall_s'] > 0
    adopted = load(family/'accepted-outcome.json')
    official = evidence/'official103'
    assert adopted['official_adopted'] and adopted['official103_accepted'] and adopted['local_GPU_pair_updated']
    assert adopted['source_count'] == 66 and adopted['GPUs'] == 2 and not adopted['website_uploaded']
    assert adopted['all_rounds_and50_steps_preserved'] and adopted['unchanged_other_video_SHA']
    assert adopted['selected_new_actual_timed_videos'] and adopted['changed_video_count'] > 0
    assert adopted['case_sum_s'] == 716.086762 and adopted['previous_case_sum_s'] == 717.513174
    assert adopted['formal_saved_s'] == 1.426412 and adopted['selected_video_count'] == 17
    assert adopted['unchanged_video_count'] == 86
    case = 'unitree_z1_dual_arm_cleanup_pencils'
    assert adopted['selected_cases'] == {case: 'nonv2_stop46_pencils_full_r1'}
    history = repo/'results/history/official-before-nonv2-stop46'
    for name, digest in adopted['assembly_script_hashes'].items():
        assert sha(official/name) == digest
    for name, digest in adopted['pair_hashes'].items():
        assert sha(repo/'results/history/official-before-dual-tail66'/name) == sha(official/'validated'/name) == digest
    for name, digest in adopted['previous_pair_hashes'].items():
        assert sha(history/name) == digest
    fresh = load(official/'validated/result.json')
    before = load(history/'result.json')
    assert fresh == load(repo/'results/history/official-before-dual-tail66/result.json') and fresh['status'] == 'accepted'
    assert fresh['psnr_validation']['video_count'] == 103 and sum(r['rounds'] for r in fresh['cases']) == 49
    for row in fresh['cases']:
        previous = next(r for r in before['cases'] if r['case'] == row['case'])
        if row['case'] == case:
            assert row['process_wall_s'] == selected['process_wall_s'] and row['psnr_db'] == selected['psnr_db']
        else:
            assert row == previous
    links = load(official/'validated/immutable-video-links.json')
    prior = load(repo/'results/aggressive-20261029/z1-reuse-window66/official103/validated/immutable-video-links.json')
    normalize = lambda row: {p.rsplit('/outputs/', 1)[1]: h for p, h in row['hashes'].items()}
    current, old = normalize(links), normalize(prior)
    assert len(current) == len(old) == 103 and current.keys() == old.keys()
    others = {n: h for n, h in current.items() if not n.startswith(case+'/')}
    assert len(others) == 86 and all(old[n] == h for n, h in others.items())
    assert {n.split('/', 1)[1]: h for n, h in current.items() if n.startswith(case+'/')} == load(evidence/'pencils-full-r1/parallel/parallel-complete.json')['video_hashes']
    assert links['ordinary_files'] and links['original_outputs_unchanged'] and links['no_GPU_rerun_or_timer_rewrite']
    print('NONV2_STOP46_ADOPTED pencils=true same_GPU_full_control=true fresh103_accepted=true other86_SHA_unchanged=true')


if __name__ == '__main__':
    verify()
