"""Verify short-window zero-slope G1 source changes and full closed-loop evidence."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    parent = repo/'experiments/g1-prediction70/deep13-r1/production-overlay'
    frozen = repo/'experiments/g1-stale70/deep13-late30-r1'
    overlay = frozen/'production-overlay'
    evidence = repo/'results/aggressive-20261029/g1-stale70'
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_native_shared_runtime68.py'
    predictor = 'src/unifolm_wma/utils/crossstep_feature_reuse.py'
    entry = 'scripts/evaluation/world_model_interaction.py'
    manifest = load(overlay/'nccl-parallel-production-sources.json')
    previous = load(parent/'nccl-parallel-production-sources.json')
    assert len(manifest) == 70 and set(manifest) == set(previous)
    assert {n for n, h in previous.items() if manifest[n] != h} == {runtime, predictor}
    for name, digest in manifest.items():
        path = overlay/name
        assert sha(path if path.exists() else base/name) == digest, name
    assets = load(frozen/'asset-hashes.json')
    for name, digest in assets.items():
        path = repo/'unifolm'/name if name.endswith('.so') else overlay/name
        assert sha(path) == digest, name
    for name, digest in load(frozen/'guard-source-hashes.json').items():
        assert sha(frozen/'guard'/name) == digest
    assert (overlay/entry).read_bytes() == (parent/entry).read_bytes().replace(b"'start': 20", b"'start': 30")
    assert (overlay/runtime).read_bytes() == (parent/runtime).read_bytes().replace(b'==10*args.n_iter', b'==5*args.n_iter')
    expected = (parent/predictor).read_text().replace('alpha=(position-last)/(last-before)', 'alpha=0.0')
    expected = expected.replace("prediction='current + (current - previous) * actual_position_gap_ratio'",
        "stale_refresh_reuse=True,prediction='latest complete refreshed feature (zero slope)'")
    expected = expected.replace('for changed in(False,True):\n                if changed:current.add_(.25);previous.mul_(.5);alpha.fill_(.5)',
        'for coefficient in(1.,.5,0.):\n                current.add_(.25);previous.mul_(.5);alpha.fill_(coefficient)')
    assert (overlay/predictor).read_text() == expected
    policy = load(frozen/'trial.json')['feature_policy']
    assert policy == dict(input_start=6, output_end=6, start=30, stop=40,
        boundary_residual=False, scope='deep', stages=['decision', 'world'])
    proof = load(overlay/'nccl-entry-timing-proof.json')
    assert proof['all_rounds_and50_DDIM_updates_preserved'] and proof['zero_slope_latest_complete_refresh']
    assert proof['original68_inference_math_and_binding_helpers_unchanged']
    assert not proof['new_native_argument_or_address_inspection']
    cpu = load(evidence/'public-repro-cpu-r1/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution'] and len(cpu['trials']) == 1
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_g1_stale70.py')
    for name, digest in cpu['logs_SHA256'].items():
        assert sha(evidence/'public-repro-cpu-r1'/name) == digest
    row = cpu['trials'][0]
    for field in ('exact_timed_manifest_and_assets', 'all1100_calls_through_executor',
                  'independent_stale_interpreter_CPU_passed', 'lexical_cache_scope_passed',
                  'full990_reuse110_call_counts_checked'):
        assert row[field]
    execution = frozen/'execution-source-identity.json'
    if execution.exists():
        identity = load(execution)
        assert identity['manifest'] == manifest and identity['asset_hashes'] == assets
        assert identity['script_hashes'] == load(frozen/'guard-source-hashes.json')
        assert identity['all_source_and_assets_SHA_unchanged'] and identity['fresh_output_and_guard_state']
        retired = load(frozen/'queued-attempt-retirement.json')
        assert retired['attempt1_contaminated_and_excluded'] and retired['attempt2_not_started']
        assert retired['no_foreign_process_signaled'] and retired['no_running_benchmark_in_owned_group']
    run = evidence/'deep13-late30-r1'
    summary = load(run/'summary.json')
    assert summary['clean'] and summary['workload_complete'] and summary['rank_videos_exact']
    assert summary['verified_source_count'] == 70 and not summary['official_adopted']
    row = next(r for r in load(run/'result.json')['cases'] if r['case'] == 'unitree_g1_pack_camera')
    assert row['rounds'] == 11 and row['videos'] == 23 and row['execution_ok'] and row['profile_complete']
    assert row['process_wall_s'] == summary['process_wall_s'] and row['psnr_db'] == summary['psnr_db']
    assert row['psnr_passed'] == summary['quality_passed'] == (row['psnr_db'] > 25)
    complete = load(run/'parallel/parallel-complete.json')
    assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal']
    assert len(complete['video_hashes']) == 23 and complete['stage_calls'] == dict(decision=550, world=550)
    for rank in (0, 1):
        ctl = load(run/f'parallel/rank{rank}-controller.json')
        assert ctl['stage_captures'] == dict(decision=31, world=31)
        controller = ctl['controller']
        assert controller['captured'] == controller['GEGLU']['captured'] == 62
        assert len(controller['native_binding']) == 2 and controller['native_linear_capture_count'] == 124
        assert controller['spatial_shared_boundary']['captured'] == 30
        assert controller['temporal_shared_boundary']['captured'] == 32
        assert {role: v['native_FFN_partitions'] for role, v in ctl['graph_roles'].items()} == dict(
            decision_full=31, decision_reuse=0, world_full=31, world_reuse=0)
        f = ctl['feature_reuse']
        assert f['stale_refresh_reuse'] and f['window'] == [30, 40]
        assert f['graph_count'] == 4 and len(f['selected_blocks']) == 13
        assert f['counts'] == dict(decision_calls=550, decision_refresh=495, decision_reuse=55,
            world_calls=550, world_refresh=495, world_reuse=55)
        assert f['alpha_histogram'] == {'0.0': 110} and f['graph_prediction_selfcheck']['passed']
        assert len(f['graph_prediction_selfcheck']['cases']) == 4
        assert len(f['completed_generations']) == 22 and all(g['steps'] == 50 for g in f['completed_generations'])
        assert not f['sampler_step_count_reduced'] and not f['round_count_reduced']
    failure = load(evidence/'preparation-failure-r1/summary.json')
    assert failure['no_GPU_execution'] and failure['CPU_preflight_caught_histogram_error']
    assert failure['corrected_before_GPU_launch']
    print('G1_STALE70_VERIFIED source70=true original_native_helpers=true full11_rounds_50steps=true time='+str(row['process_wall_s'])+' PSNR='+str(row['psnr_db']))


if __name__ == '__main__':
    verify()
