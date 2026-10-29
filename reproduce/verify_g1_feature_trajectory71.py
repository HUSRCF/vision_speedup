"""Check read-only G1 feature measurements against the exact full-refresh control."""
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    frozen = repo / 'experiments/g1-feature-trajectory71/full-r1'
    overlay = frozen / 'production-overlay'
    base = repo / 'experiments/nonv2-vae-all-display/unifolm'
    control = repo / 'experiments/g1-prediction70/refresh-control-r1/production-overlay'
    manifest = load(overlay / 'nccl-parallel-production-sources.json')
    previous = load(control / 'nccl-parallel-production-sources.json')
    observer = 'src/unifolm_wma/utils/feature_trajectory_observer71.py'
    assert len(manifest) == 71 and set(manifest) - set(previous) == {observer}
    assert all(manifest[name] == digest for name, digest in previous.items())
    for name, digest in manifest.items():
        p = overlay / name
        assert sha(p if p.exists() else base / name) == digest, name
    for name, digest in load(frozen / 'asset-hashes.json').items():
        p = repo / 'unifolm' / name if name.endswith('.so') else overlay / name
        assert sha(p) == digest, name
    for name, digest in load(frozen / 'guard-source-hashes.json').items():
        assert sha(frozen / 'guard' / name) == digest, name
    entry = 'scripts/evaluation/world_model_interaction.py'
    text = (overlay / entry).read_text()
    for line in ('        from unifolm_wma.utils import feature_trajectory_observer71 as _trajectory\n',
                 '        _trajectory.install(model)\n',
                 '    from unifolm_wma.utils import feature_trajectory_observer71 as _trajectory\n',
                 '    _trajectory.finish(args)\n'):
        assert text.count(line) == 1
        text = text.replace(line, '')
    assert text == (control / entry).read_text()
    evidence = repo / 'results/aggressive-20261029/g1-feature-trajectory71'
    cpu_root = evidence / 'public-repro-cpu-r1'
    cpu = load(cpu_root / 'summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution']
    assert cpu['public_script_SHA256'] == sha(repo / 'reproduce/run_g1_feature_trajectory71.py')
    assert cpu['trials'][0]['exact_polynomial180_CPU_checks']
    assert cpu['trials'][0]['exact_timed_manifest_and_assets']
    for name, digest in cpu['logs_SHA256'].items():
        assert sha(cpu_root / name) == digest
    run = evidence / 'full-r1'
    summary = load(run / 'summary.json')
    assert all(summary[k] for k in ('clean', 'workload_complete', 'quality_passed',
        'rank_videos_exact', 'rank_trajectory_scalar_reports_exact',
        'original68_23_videos_SHA256_equal', 'diagnostic_only'))
    assert not summary['official_adopted'] and summary['gpu_candidates'] == ['0', '1']
    complete = load(run / 'parallel/parallel-complete.json')
    original = load(repo / 'results/aggressive-20261029/half-native-shared68/full-r2/parallel/parallel-complete.json')
    assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal']
    assert len(complete['video_hashes']) == 23 and complete['video_hashes'] == original['video_hashes']
    assert complete['stage_calls'] == dict(decision=550, world=550)
    reports = [load(run / f'parallel/rank{rank}-feature-trajectory.json') for rank in (0, 1)]
    assert reports[0] == reports[1]
    r = reports[0]
    for key in ('read_only_feature_trajectory_observer', 'full_refresh_reference_only',
                'no_features_or_weights_exported', 'no_new_native_argument_or_address_inspection',
                'all50_steps_and11_rounds_preserved', 'inference_outputs_returned_unchanged',
                'offline_errors_do_not_include_approximate_feedback'):
        assert r[key]
    assert r['primitive_GPU_selfcheck']['passed'] and len(r['primitive_GPU_selfcheck']['cases']) == 4
    assert len(r['completed_generations']) == 22 and all(x['slots'] == 50 and x['all_finite'] for x in r['completed_generations'])
    assert len(r['tables']) == 3 and r['persistent_feature_bytes'] == 5898240000
    assert sum(x['bytes'] for x in r['tables'].values()) == r['persistent_feature_bytes']
    assert len(r['statistics']) == 5940
    groups = defaultdict(list)
    for row in r['statistics']:
        start, stop = row['window']
        assert start <= row['position'] < stop and (row['position'] - start) % 2 == 1
        old, previous, last = row['history']
        assert old < previous < last < row['position'] and row['order'] in (0, 1, 2)
        assert all(math.isfinite(row[k]) and row[k] >= 0 for k in ('mse', 'relative_l2', 'max_abs'))
        groups[row['stage'], row['scope'], tuple(row['window']), row['order']].append(row)
    derived = []
    for (stage, scope, window, order), rows in sorted(groups.items()):
        assert len(rows) == 11 * ((window[1] - window[0]) // 2)
        derived.append(dict(stage=stage, scope=scope, window=list(window), order=order,
            samples=len(rows), mean_mse=statistics.mean(x['mse'] for x in rows),
            median_relative_l2=statistics.median(x['relative_l2'] for x in rows),
            max_relative_l2=max(x['relative_l2'] for x in rows)))
    assert derived == load(run / 'scalar-summary.json')['groups']
    for rank in (0, 1):
        ctl = load(run / f'parallel/rank{rank}-controller.json')
        assert set(ctl['graph_roles']) == {'decision_full', 'world_full'}
        assert ctl['feature_reuse']['counts'] == dict(decision_calls=550, decision_refresh=550, world_calls=550, world_refresh=550)
        assert ctl['feature_reuse']['graph_count'] == 2 and ctl['feature_reuse']['stages'] == []
    adopted = load(repo / 'experiments/nonv2-late23-prediction65/dual-accepted-outcome.json')
    for name, digest in adopted['pair_hashes'].items():
        assert sha(repo / 'results/history/official-before-resblock-tail66' / name) == digest
    print('G1_TRAJECTORY71_VERIFIED original70_sources_exact=true full23_video_SHA_exact=true scalar5940_each_rank=true no_approximate_feedback=true formal_unchanged=true')


if __name__ == '__main__':
    verify()
