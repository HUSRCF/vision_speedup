"""Verify deep13 predictor integration without inheriting single-rank quality."""
import ast
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    frozen = repo/'experiments/nonv2-prediction65/full-r2'
    first = repo/'experiments/nonv2-prediction65/full-r1'
    overlay = frozen/'production-overlay'
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    original = load(repo/'experiments/nonv2-all-blocks59/full63-r1/production-sources.json')
    manifest = load(frozen/'production-sources.json')
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    graph = 'src/unifolm_wma/utils/cuda_graph.py'
    feature = 'src/unifolm_wma/utils/crossstep_feature_runtime.py'
    reuse = 'src/unifolm_wma/utils/crossstep_feature_reuse.py'
    assert len(manifest) == 65
    assert {n for n,h in original.items() if manifest[n] != h} == {runtime,graph}
    assert set(manifest)-set(original) == {feature,reuse}
    for name,digest in manifest.items():
        p = overlay/name
        assert sha(p if p.exists() else base/name) == digest, name
    for name,digest in load(frozen/'timing-asset-hashes.json').items():
        assert sha(overlay/name) == digest, name
    for name,digest in load(frozen/'guard-source-hashes.json').items():
        assert sha(frozen/name) == digest, name
    for name in ('production-sources.json','timing-asset-hashes.json','guard-source-hashes.json','flags.json','timing-proof.json','trial.json'):
        assert (first/name).read_bytes() == (frozen/name).read_bytes(), name
    assert load(frozen/'flags.json') == load(repo/'experiments/nonv2-all-blocks59/full63-r1/flags.json')
    trial = load(frozen/'trial.json')
    expected = dict(decision_full=33,decision_reuse=21,world_full=33,world_reuse=21)
    assert trial['per_role_expected_gathers'] == expected
    assert trial['rounds'] == 12 and trial['videos'] == 25 and trial['GPUs'] == 2
    assert trial['prediction_policy']['history'] == 2 and trial['cache_scope_isolated']
    single = repo/'experiments/temporal-prediction61/deep13-r1/production-overlay'
    nodes = lambda p: {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old = nodes(single/reuse)
    new = nodes(overlay/reuse)
    for name in ('_predict_linear','graph_check','Controller'):
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    predicted = new['predict']
    locations = [i for i,n in enumerate(predicted.body) if isinstance(n,ast.With)]
    assert len(locations) == 1
    index = locations[0]
    assert ast.unparse(predicted.body[index].items[0].context_expr) == 'kernel_cache_scope()'
    predicted.body[index:index+1] = predicted.body[index].body
    assert ast.dump(predicted) == ast.dump(old['predict'])
    static = repo/'experiments/nonv2-static-timestep65/full-r4/production-overlay/src/unifolm_wma/utils/static_timestep_cache.py'
    previous_scope = ast.unparse(nodes(static)['kernel_cache_scope']).replace('TEAM07_STATIC_TIME_CACHE','TEAM07_PREDICT_CACHE')
    assert ast.unparse(new['kernel_cache_scope']) == previous_scope
    cpu = (frozen/'crossstep-cpu.log').read_text()
    assert 'all1200_calls_through_parallel_executor=true' in cpu
    assert 'affine_complete_outputs=3600' in cpu
    assert 'PREDICT_CACHE_SCOPE_CPU passed=true environment_and_Triton_class_restored=true original_FFN_assertion_retained=true' in cpu
    public = load(repo/'results/aggressive-20261029/nonv2-prediction65/public-repro-cpu-r1/summary.json')
    assert public['passed'] and public['no_GPU_execution'] and public['exact_timed_manifest_and_assets']
    assert public['all1200_calls_through_parallel_executor'] and public['predictor_affine3600_CPU_passed'] and public['lexical_cache_scope_passed']
    assert public['public_script_SHA256'] == sha(repo/'reproduce/run_nonv2_prediction65.py')
    evidence = repo/'results/aggressive-20261029/nonv2-prediction65'
    wait = load(evidence/'full-r1/summary.json')
    assert wait['wait_only_retired'] and not wait['clean'] and wait['process_wall_s'] is None and wait['psnr_db'] is None
    retired = load(evidence/'full-r1/waiting-reservation-retired.json')
    assert retired['no_benchmark_started'] and retired['no_speed_or_quality_claim']
    full = evidence/'full-r2'
    summary = load(full/'summary.json')
    assert summary['clean'] and summary['workload_complete'] and summary['rank_videos_exact']
    row = next(r for r in load(full/'result.json')['cases'] if r['case'] == 'unitree_z1_stackbox')
    assert row['process_wall_s'] == summary['process_wall_s'] and row['psnr_db'] == summary['psnr_db']
    assert row['psnr_passed'] == summary['quality_passed'] and row['rounds'] == 12 and row['videos'] == 25
    complete = load(full/'parallel/parallel-complete.json')
    assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal'] and len(complete['video_hashes']) == 25
    assert complete['stage_calls'] == dict(decision=600,world=600)
    for rank in (0,1):
        controller = load(full/f'parallel/rank{rank}-controller.json')
        assert controller['stage_captures'] == dict(decision=54,world=54) and controller['controller']['captured'] == 108
        assert {role:len(names) for role,names in controller['graph_role_modules'].items()} == expected
        report = controller['feature_reuse']
        assert report['predictive_feature_reuse'] and report['graph_count'] == 4
        assert report['alpha_histogram'] == {'1.0':24,'0.5':336}
        assert len(report['completed_generations']) == 24 and len(report['selected_blocks']) == 13
        assert report['counts']['decision_calls'] == report['counts']['world_calls'] == 600
        assert report['counts']['decision_reuse'] == report['counts']['world_reuse'] == 180
        assert not report['round_count_reduced'] and not report['sampler_step_count_reduced']
        assert report['graph_prediction_selfcheck']['passed']
    accepted = load(repo/'experiments/nonv2-prediction65/accepted-outcome.json')
    assert accepted['official_adopted'] and accepted['official103_accepted'] and accepted['local_GPU_pair_updated']
    assert accepted['Z1_s'] == row['process_wall_s'] == 145.327923
    assert accepted['PSNR_dB'] == row['psnr_db'] == 28.992932876074295
    assert accepted['case_sum_s'] == 758.855686 and accepted['formal_saved_s'] == 5.510466
    assert accepted['historical_comparison_not_matched_AB'] and not accepted['website_uploaded']
    fresh = load(evidence/'official103/validated/result.json')
    assert fresh == load(repo/'results/history/official-before-other-prediction65/result.json') and fresh['status'] == 'accepted'
    assert fresh['psnr_validation']['video_count'] == 103
    history = repo/'results/history/official-before-prediction65'
    for name,digest in accepted['pair_hashes'].items():
        assert sha(repo/'results/history/official-before-other-prediction65'/name) == sha(evidence/'official103/validated'/name) == digest
    for name,digest in accepted['previous_pair_hashes'].items():
        assert sha(history/name) == digest
    before = load(history/'result.json')
    for current in fresh['cases']:
        previous = next(r for r in before['cases'] if r['case'] == current['case'])
        if current['case'] == 'unitree_z1_stackbox':
            assert current['process_wall_s'] == row['process_wall_s'] and current['psnr_db'] == row['psnr_db']
        else:
            assert current == previous
    copies = load(evidence/'official103/validated/immutable-video-copies.json')
    prior = load(repo/'results/aggressive-20261029/nonv2-crossstep65/official103/validated/immutable-video-copies.json')
    others = lambda data: {p:h for p,h in data['hashes'].items() if '/unitree_z1_stackbox/' not in p}
    assert copies['count'] == 103 and copies['ordinary_files'] and copies['original_outputs_unchanged']
    assert len(others(copies)) == 78 and others(copies) == others(prior)
    print('Adopted prediction65: unchanged predictor arithmetic,scoped-cache restoration,full12/25 two-rank quality and fresh103 with unchanged other78 videos verified.')


if __name__ == '__main__':
    verify()
