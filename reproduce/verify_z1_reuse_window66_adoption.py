"""Verify the chosen stop46 full comparison and newly accepted103 aggregate."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    family = repo/'experiments/z1-reuse-window66'
    evidence = repo/'results/aggressive-20261029/z1-reuse-window66'
    candidate = repo/'results/aggressive-20261029/z1-tailwindow23-tail66/full-r1'
    baseline = repo/'results/aggressive-20261029/z1-window23-tail66/start16-baseline-r1'
    selected = load(candidate/'summary.json'); control = load(baseline/'summary.json')
    for row in (selected, control):
        assert row['clean'] and row['workload_complete'] and row['quality_passed'] and row['rank_videos_exact']
        assert row['gpu_candidates'] == ['0', '2'] and row['verified_source_count'] == 66
    assert selected['process_wall_s'] == 127.443856 and selected['psnr_db'] == 32.547700729334316
    assert control['psnr_db'] == 30.332499792023174 and control['process_wall_s'] > selected['process_wall_s']
    identity = load(repo/'experiments/z1-window23-tail66/start16-baseline-r1/source-identity.json')
    parent = repo/'experiments/z1-late23-tail66/full-r1'
    assert identity['manifest'] == load(parent/'production-sources.json')
    assert identity['asset_hashes'] == load(parent/'timing-asset-hashes.json')
    control_hashes = load(baseline/'parallel/parallel-complete.json')['video_hashes']
    assert control_hashes == load(repo/'results/aggressive-20261029/z1-late23-tail66/full-r2/parallel/parallel-complete.json')['video_hashes']
    comparison = load(evidence/'complete-process-comparison.json')
    assert comparison['selected_candidate'] == 'stop46' and comparison['GPUs'] == ['0', '2']
    assert comparison['selected_candidate_immediately_before_baseline'] and not comparison['multiple_independent_matched_pairs']
    assert comparison['candidate_s'] == selected['process_wall_s'] and comparison['baseline_s'] == control['process_wall_s']
    assert comparison['saved_s'] == control['process_wall_s']-selected['process_wall_s'] > 0
    adopted = load(family/'accepted-outcome.json'); official = evidence/'official103'
    assert adopted['official_adopted'] and adopted['official103_accepted'] and adopted['local_GPU_pair_updated']
    assert adopted['source_count'] == 66 and adopted['GPUs'] == 2 and not adopted['website_uploaded']
    assert adopted['all_rounds_and50_steps_preserved'] and adopted['unchanged_other78_video_SHA']
    assert adopted['selected25_new_actual_timed_videos'] and adopted['changed_video_count'] == 25
    assert adopted['case_sum_s'] == 717.513174 and adopted['previous_case_sum_s'] == 722.609176
    assert adopted['formal_saved_s'] == 5.096002 and adopted['selected_video_count'] == 25 and adopted['unchanged_video_count'] == 78
    assert set(adopted['selected_cases']) == {'unitree_z1_stackbox'}
    assert set(adopted['selected_cases'].values()) == {'z1_tailwindow23_tail66_full_r1'}
    history = repo/'results/history/official-before-z1-reuse-window66'
    for name, h in adopted['assembly_script_hashes'].items(): assert sha(official/name) == h
    for name, h in adopted['pair_hashes'].items(): assert sha(repo/'results/history/official-before-nonv2-stop46'/name) == sha(official/'validated'/name) == h
    for name, h in adopted['previous_pair_hashes'].items(): assert sha(history/name) == h
    fresh = load(official/'validated/result.json'); before = load(history/'result.json')
    assert fresh == load(repo/'results/history/official-before-nonv2-stop46/result.json') and fresh['status'] == 'accepted'
    assert fresh['psnr_validation']['video_count'] == 103 and sum(x['rounds'] for x in fresh['cases']) == 49
    for row in fresh['cases']:
        previous = next(x for x in before['cases'] if x['case'] == row['case'])
        if row['case'] == 'unitree_z1_stackbox':
            assert row['process_wall_s'] == selected['process_wall_s'] and row['psnr_db'] == selected['psnr_db']
        else: assert row == previous
    links = load(official/'validated/immutable-video-links.json')
    prior = load(repo/'results/aggressive-20261029/z1-late23-tail66/official103/validated/immutable-video-links.json')
    normalize = lambda row: {p.rsplit('/outputs/', 1)[1]: h for p, h in row['hashes'].items()}
    current, old = normalize(links), normalize(prior)
    assert len(current) == len(old) == 103 and current.keys() == old.keys()
    others = {n: h for n, h in current.items() if not n.startswith('unitree_z1_stackbox/')}
    assert len(others) == 78 and all(old[n] == h for n, h in others.items())
    assert {n.split('/', 1)[1]: h for n, h in current.items() if n.startswith('unitree_z1_stackbox/')} == load(candidate/'parallel/parallel-complete.json')['video_hashes']
    assert links['ordinary_files'] and links['original_outputs_unchanged'] and links['no_GPU_rerun_or_timer_rewrite']
    print('Z1_REUSE_WINDOW66_ADOPTED stop46=true same_GPU_full_comparison=true full50steps=true fresh103_accepted=true other78_SHA_unchanged=true')


if __name__ == '__main__': verify()
