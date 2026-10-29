"""Verify cadence plumbing, unchanged prediction math and independent full quality."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    family = repo/'experiments/z1-interval3-primed66/full-r1'
    evidence = repo/'results/aggressive-20261029/z1-interval3-primed66'
    parent = repo/'experiments/z1-tailwindow23-tail66/full-r1'
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    feature_name = 'src/unifolm_wma/utils/crossstep_feature_runtime.py'
    reuse = 'src/unifolm_wma/utils/crossstep_feature_reuse.py'
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    entry = 'scripts/evaluation/world_model_interaction.py'
    overlay = family/'production-overlay'
    manifest = load(family/'production-sources.json'); old = load(parent/'production-sources.json')
    assert set(manifest) == set(old) and len(manifest) == 66
    assert {n for n, h in old.items() if manifest[n] != h} == {feature_name, reuse, runtime}
    for name, h in manifest.items():
        p = overlay/name
        assert sha(p if p.exists() else base/name) == h, name
    assets = load(family/'timing-asset-hashes.json'); old_assets = load(parent/'timing-asset-hashes.json')
    assert {n for n, h in old_assets.items() if assets[n] != h} == {entry, 'nccl-entry-timing-proof.json'}
    for name, h in assets.items(): assert sha(overlay/name) == h
    for name, h in load(family/'guard-source-hashes.json').items(): assert sha(family/name) == h
    before = (parent/'production-overlay'/feature_name).read_text()
    for old_text, new_text in [
        ('output_end=6,start=10,stop=40,boundary_residual=False', 'output_end=6,start=10,stop=40,interval=2,boundary_residual=False'),
        ('output_end=output_end,start=start,stop=stop,boundary_residual=', 'output_end=output_end,start=start,stop=stop,interval=interval,boundary_residual='),
        ("interval=2,stages=list(options.get('stages'", "interval=options.get('interval',2),stages=list(options.get('stages'")]:
        assert before.count(old_text) == 1; before = before.replace(old_text, new_text)
    assert (overlay/feature_name).read_text() == before
    before = (parent/'production-overlay'/reuse).read_text()
    before = before.replace('for changed in(False,True):\n                if changed:current.add_(.25);previous.mul_(.5);alpha.fill_(.5)',
                            'for scalar in(1.,.5,1/3,2/3,2.):\n                current.add_(.25);previous.mul_(.5);alpha.fill_(scalar)')
    before = before.replace('        alpha=0.\n', '        reuse=reuse and not(self.interval==3 and position==self.start+1)\n        alpha=0.\n')
    before = before.replace('tail=257,mutated_replay_bitwise=True', 'tail=257,mutated_replay_bitwise=True,alphas=[1.,.5,1/3,2/3,2.]')
    assert (overlay/reuse).read_text() == before
    before = (parent/'production-overlay'/runtime).read_text()
    assert (overlay/runtime).read_text() == before.replace('==13*args.n_iter', '==16*args.n_iter')
    before = (parent/'production-overlay'/entry).read_text()
    assert before.count("'stages': ('decision', 'world')}") == 1
    assert (overlay/entry).read_text() == before.replace("'stages': ('decision', 'world')}", "'stages': ('decision', 'world'), 'interval': 3}")
    trial = load(family/'trial.json')
    assert trial['feature_policy'] == dict(input_start=1, output_end=11, start=20, stop=46,
                                          boundary_residual=False, scope='deep', stages=['decision', 'world'], interval=3)
    assert trial['rounds'] == 12 and trial['videos'] == 25 and trial['GPUs'] == 2
    cpu = load(evidence/'public-repro-cpu-r1/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution'] and len(cpu['trials']) == 1
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_z1_interval3_primed66.py')
    for name, h in cpu['logs_SHA256'].items(): assert sha(evidence/'public-repro-cpu-r1'/name) == h
    text = (evidence/'public-repro-cpu-r1/z1-check_nonv2_crossstep65_cpu.py.log').read_text()
    assert 'Z1_INTERVAL3_PRIMED66_CPU passed=true actual23_parents=true all1200_affine_outputs_exact=true' in text
    assert 'interval3=true' in text and 'PREDICT_CACHE_SCOPE_CPU passed=true' in text and 'affine_complete_outputs=3600' in text
    run = evidence/'full-r1'; summary = load(run/'summary.json')
    assert summary['clean'] and summary['workload_complete'] and summary['rank_videos_exact']
    assert summary['verified_source_count'] == 66 and not summary['official_adopted']
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
        assert feature['interval'] == 3 and feature['window'] == [20, 46] and len(feature['selected_blocks']) == 23
        assert feature['graph_count'] == 4 and feature['graph_prediction_selfcheck']['passed']
        cases = feature['graph_prediction_selfcheck']['cases']; assert len(cases) == 4
        assert all(x['alphas'] == [1., .5, 1/3, 2/3, 2.] and x['mutated_replay_bitwise'] and x['tail'] == 257 for x in cases)
        assert feature['counts'] == dict(decision_calls=600, decision_refresh=408, decision_reuse=192,
                                         world_calls=600, world_refresh=408, world_reuse=192)
        assert feature['alpha_histogram'] == {'1.0': 48, '0.5': 24, str(1/3): 168, str(2/3): 144}
        assert not feature['round_count_reduced'] and not feature['sampler_step_count_reduced']
    print('Z1_INTERVAL3_PRIMED66_VERIFIED original_prediction_math=true five_GPU_scalars=true full12_rounds_50steps=true PSNR=' + str(summary['psnr_db']))


if __name__ == '__main__': verify()
