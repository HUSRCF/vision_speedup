"""Verify wider-window Z1 trials against the accepted late23 math and full videos."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    family = repo/'experiments/z1-tailwindow23-tail66'
    evidence = repo/'results/aggressive-20261029/z1-tailwindow23-tail66'
    parent = repo/'experiments/z1-late23-tail66/full-r1'
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    old = load(parent/'production-sources.json'); old_assets = load(parent/'timing-asset-hashes.json')
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    entry = 'scripts/evaluation/world_model_interaction.py'
    trials = []
    for frozen in [family/'full-r1']:
        start = 20; stop = 46
        predictions = (stop-start)//2; overlay = frozen/'production-overlay'
        manifest = load(frozen/'production-sources.json'); assets = load(frozen/'timing-asset-hashes.json')
        assert set(manifest) == set(old) and len(manifest) == 66
        assert {n for n, h in old.items() if manifest[n] != h} == {runtime}
        assert {n for n, h in old_assets.items() if assets[n] != h} == {entry, 'nccl-entry-timing-proof.json'}
        for name, h in manifest.items():
            path = overlay/name
            assert sha(path if path.exists() else base/name) == h
        for name, h in assets.items(): assert sha(overlay/name) == h
        for name, h in load(frozen/'guard-source-hashes.json').items(): assert sha(frozen/name) == h
        before = (parent/'production-overlay'/entry).read_text()
        assert before.count("'stop': 40") == 1
        assert (overlay/entry).read_text() == before.replace("'stop': 40", "'stop': 46")
        before = (parent/'production-overlay'/runtime).read_text()
        assert before.count('==10*args.n_iter') == 1
        assert (overlay/runtime).read_text() == before.replace('==10*args.n_iter', f'=={predictions}*args.n_iter')
        trial = load(frozen/'trial.json')
        assert trial['feature_policy'] == dict(input_start=1, output_end=11, start=start, stop=stop,
                                              boundary_residual=False, scope='deep', stages=['decision', 'world'])
        assert trial['rounds'] == 12 and trial['videos'] == 25 and trial['GPUs'] == 2
        assert trial['per_role_expected_gathers'] == dict(decision_full=38, decision_reuse=4, world_full=38, world_reuse=4)
        cpu = load(evidence/'public-repro-cpu-r1/summary.json')
        assert cpu['passed'] and cpu['no_GPU_execution'] and len(cpu['trials']) == 1
        assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_z1_tailwindow23_tail66.py')
        for name, h in cpu['logs_SHA256'].items(): assert sha(evidence/'public-repro-cpu-r1'/name) == h
        text = (evidence/'public-repro-cpu-r1/z1-check_nonv2_crossstep65_cpu.py.log').read_text()
        assert 'Z1_TAILWINDOW23_TAIL66_CPU passed=true actual23_parents=true all1200_affine_outputs_exact=true' in text
        assert 'window20_46=true' in text and 'PREDICT_CACHE_SCOPE_CPU passed=true' in text
        run = evidence/frozen.name; summary = load(run/'summary.json')
        assert summary['clean'] and summary['workload_complete'] and summary['rank_videos_exact']
        assert summary['verified_source_count'] == 66 and not summary['official_adopted']
        row = next(x for x in load(run/'result.json')['cases'] if x['case'] == 'unitree_z1_stackbox')
        assert row['rounds'] == 12 and row['videos'] == 25 and row['execution_ok'] and row['profile_complete']
        assert row['process_wall_s'] == summary['process_wall_s'] and row['psnr_db'] == summary['psnr_db']
        assert summary['quality_passed'] == row['psnr_passed'] == (row['psnr_db'] > 25)
        complete = load(run/'parallel/parallel-complete.json')
        assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal'] and len(complete['video_hashes']) == 25
        assert complete['stage_calls'] == dict(decision=600, world=600)
        for rank in (0, 1):
            ctl = load(run/f'parallel/rank{rank}-controller.json')
            assert ctl['stage_captures'] == dict(decision=36, world=36) and ctl['controller']['captured'] == 72
            assert ctl['controller']['resblock_frames']['captured'] == 12
            assert {k: len(v) for k, v in ctl['graph_role_modules'].items()} == dict(decision_full=33, decision_reuse=3, world_full=33, world_reuse=3)
            assert {k: len(v) for k, v in ctl['resblock_graph_role_modules'].items()} == dict(decision_full=5, decision_reuse=1, world_full=5, world_reuse=1)
            feature = ctl['feature_reuse']; uses = 12*predictions
            assert feature['window'] == [start, stop] and len(feature['selected_blocks']) == 23
            assert feature['graph_count'] == 4 and feature['graph_prediction_selfcheck']['passed']
            assert feature['counts'] == dict(decision_calls=600, decision_refresh=600-uses, decision_reuse=uses,
                                             world_calls=600, world_refresh=600-uses, world_reuse=uses)
            assert feature['alpha_histogram'] == {'1.0': 24, '0.5': 24*(predictions-1)}
            assert not feature['round_count_reduced'] and not feature['sampler_step_count_reduced']
        trials.append((start, summary))
    assert trials
    print('Z1_TAILWINDOW23_TAIL66_VERIFIED source66=true original_math=true full12_rounds_50steps=true results=' + str([(s, r['process_wall_s'], r['psnr_db']) for s, r in trials]))


if __name__ == '__main__': verify()
