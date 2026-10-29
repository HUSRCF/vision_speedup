"""Check exact per-case source changes and independently measured full videos."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    family = repo/'experiments/nonv2-stop46'
    evidence = repo/'results/aggressive-20261029/nonv2-stop46'
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    entry = 'scripts/evaluation/world_model_interaction.py'
    cpu = load(evidence/'public-repro-cpu-r1/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution'] and len(cpu['trials']) == 2
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_nonv2_stop46.py')
    for name, digest in cpu['logs_SHA256'].items():
        assert sha(evidence/'public-repro-cpu-r1'/name) == digest
    measured = []
    for alias, parent_name, rounds, count, tail in (
        ('pencils', 'nonv2-resblock-tail73/full66-r1', 8, 66, True),
        ('dual', 'nonv2-late23-prediction65/dual-full-r1', 7, 65, False),
    ):
        frozen = family/f'{alias}-full-r1'
        parent = repo/'experiments'/parent_name
        overlay = frozen/'production-overlay'
        manifest = load(frozen/'production-sources.json')
        old = load(parent/'production-sources.json')
        assets = load(frozen/'timing-asset-hashes.json')
        old_assets = load(parent/'timing-asset-hashes.json')
        assert len(manifest) == count and set(manifest) == set(old)
        assert {n for n, h in old.items() if manifest[n] != h} == {runtime}
        assert {n for n, h in old_assets.items() if assets[n] != h} == {entry, 'nccl-entry-timing-proof.json'}
        for name, digest in manifest.items():
            path = overlay/name
            assert sha(path if path.exists() else base/name) == digest, name
        for name, digest in assets.items():
            assert sha(overlay/name) == digest, name
        for name, digest in load(frozen/'guard-source-hashes.json').items():
            assert sha(frozen/name) == digest, name
        before = (parent/'production-overlay'/entry).read_bytes()
        assert before.count(b"'stop': 40") == 1
        assert (overlay/entry).read_bytes() == before.replace(b"'stop': 40", b"'stop': 46")
        before = (parent/'production-overlay'/runtime).read_bytes()
        assert before.count(b'==10*args.n_iter') == 1
        assert (overlay/runtime).read_bytes() == before.replace(b'==10*args.n_iter', b'==13*args.n_iter')
        trial = load(frozen/'trial.json')
        assert trial['rounds'] == rounds and trial['videos'] == 2*rounds+1 and trial['GPUs'] == 2
        assert trial['source_count'] == count and trial['existing_tail5_sharding'] == tail
        assert trial['feature_policy'] == dict(input_start=1, output_end=11, start=20, stop=46,
            boundary_residual=False, scope='deep', stages=['decision', 'world'])
        text = (evidence/f'public-repro-cpu-r1/{alias}-check_nonv2_crossstep65_cpu.py.log').read_text()
        assert f'NONV2_STOP46_CPU passed=true actual23_parents=true all{100*rounds}_affine_outputs_exact=true' in text
        assert 'window20_46=true' in text and 'PREDICT_CACHE_SCOPE_CPU passed=true' in text
        label = f'{alias}-full-r1' if alias == 'pencils' else 'dual-full-r2'
        if alias == 'dual':
            identity = load(family/label/'source-identity.json')
            assert identity['manifest'] == manifest and identity['asset_hashes'] == assets
            assert identity['script_hashes'] == load(frozen/'guard-source-hashes.json')
            assert identity['all_source_and_assets_SHA_unchanged'] and identity['fresh_output_and_guard_state']
            retirement = load(frozen/'queued-attempt-retirement.json')
            assert retirement['initial_attempt_never_started']
            assert retirement['no_foreign_process_signaled'] and retirement['no_running_benchmark_in_owned_group']
        run = evidence/label
        summary = load(run/'summary.json')
        assert summary['clean'] and summary['workload_complete'] and summary['rank_videos_exact']
        assert summary['verified_source_count'] == count and not summary['official_adopted']
        row = next(r for r in load(run/'result.json')['cases'] if r['case'] == trial['case'])
        assert row['rounds'] == rounds and row['videos'] == 2*rounds+1
        assert row['execution_ok'] and row['profile_complete']
        assert row['process_wall_s'] == summary['process_wall_s'] and row['psnr_db'] == summary['psnr_db']
        assert summary['quality_passed'] == row['psnr_passed'] == (row['psnr_db'] > 25)
        complete = load(run/'parallel/parallel-complete.json')
        assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal']
        assert len(complete['video_hashes']) == 2*rounds+1
        assert complete['stage_calls'] == dict(decision=50*rounds, world=50*rounds)
        for rank in (0, 1):
            ctl = load(run/f'parallel/rank{rank}-controller.json')
            assert ctl['stage_captures'] == dict(decision=36, world=36)
            assert ctl['controller']['captured'] == 72
            assert {k: len(v) for k, v in ctl['graph_role_modules'].items()} == dict(
                decision_full=33, decision_reuse=3, world_full=33, world_reuse=3)
            if tail:
                assert ctl['controller']['resblock_frames']['captured'] == 12
                assert {k: len(v) for k, v in ctl['resblock_graph_role_modules'].items()} == dict(
                    decision_full=5, decision_reuse=1, world_full=5, world_reuse=1)
            feature = ctl['feature_reuse']
            assert feature['window'] == [20, 46] and len(feature['selected_blocks']) == 23
            assert feature['graph_count'] == 4 and feature['graph_prediction_selfcheck']['passed']
            assert feature['history_complete_refreshes'] == 2
            assert feature['counts'] == dict(decision_calls=50*rounds, decision_refresh=37*rounds,
                decision_reuse=13*rounds, world_calls=50*rounds, world_refresh=37*rounds,
                world_reuse=13*rounds)
            assert feature['alpha_histogram'] == {'1.0': 2*rounds, '0.5': 24*rounds}
            assert len(feature['completed_generations']) == 2*rounds
            assert all(g['steps'] == 50 for g in feature['completed_generations'])
            assert not feature['round_count_reduced'] and not feature['sampler_step_count_reduced']
        measured.append((alias, row['process_wall_s'], row['psnr_db']))
    print('NONV2_STOP46_VERIFIED exact_per_case_sources=true dual_GPU=true all_rounds_and50steps=true results='+str(measured))


if __name__ == '__main__':
    verify()
