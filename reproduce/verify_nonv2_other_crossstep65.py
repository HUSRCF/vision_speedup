"""Verify exact two-GPU dual-arm/pencils reuse trials and full-run evidence."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    load = lambda path: json.loads(path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    graph = 'src/unifolm_wma/utils/cuda_graph.py'
    feature = 'src/unifolm_wma/utils/crossstep_feature_runtime.py'
    reuse = 'src/unifolm_wma/utils/crossstep_feature_reuse.py'
    z1 = repo/'experiments/nonv2-crossstep65/full-r1/production-overlay'
    evidence = repo/'results/aggressive-20261029/nonv2-other-crossstep65'
    for alias, case, rounds in (
        ('dual', 'unitree_z1_dual_arm_stackbox', 7),
        ('pencils', 'unitree_z1_dual_arm_cleanup_pencils', 8),
    ):
        frozen = repo/'experiments/nonv2-other-crossstep65'/(alias+'-full-r1')
        overlay = frozen/'production-overlay'
        prior = repo/'experiments/nonv2-other-blocks63'/(alias+'-full-r1')
        original = load(prior/'production-sources.json')
        manifest = load(frozen/'production-sources.json')
        assert len(manifest) == 65
        assert {n for n,h in original.items() if manifest[n] != h} == {runtime, graph}
        assert set(manifest)-set(original) == {feature, reuse}
        for name, digest in manifest.items():
            path = overlay/name
            assert sha(path if path.exists() else base/name) == digest, name
        for name, digest in load(frozen/'timing-asset-hashes.json').items():
            assert sha(overlay/name) == digest, name
        for name, digest in load(frozen/'guard-source-hashes.json').items():
            assert sha(frozen/name) == digest, name
        assert load(frozen/'flags.json') == load(prior/'flags.json')
        trial = load(frozen/'trial.json')
        assert (trial['case'], trial['rounds'], trial['videos'], trial['GPUs']) == (case, rounds, 2*rounds+1, 2)
        expected = dict(decision_full=33, decision_reuse=31, world_full=33, world_reuse=31)
        assert trial['per_role_expected_gathers'] == expected
        assert sha(overlay/reuse) == sha(z1/reuse)
        assert sha(overlay/feature) == sha(z1/feature)
        entry = (overlay/'scripts/evaluation/world_model_interaction.py').read_text()
        assert entry.index('PROCESS_START_TIME =') < entry.index('_nccl_early_start()') < entry.index('_feature_runtime.install(model,')
        assert entry.index('_feature_runtime.finish(args)') < entry.index('_nccl_finish(args)') < entry.index('timing.write(args.profile_path)')
        source = (overlay/runtime).read_text()
        assert "assert _graph_captures==expected" in source
        assert 'with _mode(True):result=original(*args,**kwargs)' in source
        assert 'len(_denoiser_graph.entries)==4' in source
        cpu = load(evidence/(alias+'-public-repro-cpu-r1')/'summary.json')
        assert cpu['passed'] and cpu['no_GPU_execution'] and cpu['source_count'] == 65
        assert cpu['exact_timed_manifest_and_assets'] and cpu['all1200_calls_through_parallel_executor']
        assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_nonv2_other_crossstep65.py')
        full = evidence/(alias+'-full-r1')
        summary = load(full/'summary.json')
        assert summary['case'] == case and summary['rounds'] == rounds and summary['videos'] == 2*rounds+1
        assert summary['clean'] and summary['workload_complete'] and summary['rank_videos_exact']
        row = next(r for r in load(full/'result.json')['cases'] if r['case'] == case)
        assert row['process_wall_s'] == summary['process_wall_s']
        assert row['psnr_db'] == summary['psnr_db'] and row['psnr_passed'] == summary['quality_passed']
        assert row['rounds'] == rounds and row['videos'] == 2*rounds+1
        report = load(full/'nccl-full-report.json')
        assert report['clean'] and report['physical_GPUs'] == summary['GPUs']
        complete = load(full/'parallel/parallel-complete.json')
        assert complete['GPUs'] == 2 and complete['rank_videos_SHA256_equal']
        assert complete['stage_calls'] == dict(decision=50*rounds, world=50*rounds)
        assert len(complete['video_hashes']) == 2*rounds+1
        for rank in (0, 1):
            controller = load(full/f'parallel/rank{rank}-controller.json')
            assert controller['stage_captures'] == dict(decision=64, world=64)
            assert controller['controller']['captured'] == 128
            assert {role:len(names) for role,names in controller['graph_role_modules'].items()} == expected
            reuse_report = controller['feature_reuse']
            assert reuse_report['graph_count'] == 4
            assert len(reuse_report['completed_generations']) == 2*rounds
            assert reuse_report['counts']['decision_calls'] == reuse_report['counts']['world_calls'] == 50*rounds
            assert reuse_report['counts']['decision_reuse'] == reuse_report['counts']['world_reuse'] == 15*rounds
            assert not reuse_report['round_count_reduced'] and not reuse_report['sampler_step_count_reduced']
    print('Two nonV2 cases: exact65 source rebuilds, full two-rank workloads,128 graph-role captures and independent quality evidence verified.')


if __name__ == '__main__':
    verify()
