"""Verify experiment identity and distinguish CPU preparation from GPU evidence."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    prior = repo/'experiments/nonv2-all-blocks59/world-model-r1/scripts'
    family = repo/'experiments/nonv2-temporal-projection69'
    evidence = repo/'results/aggressive-20261029/nonv2-temporal-projection69'
    cpu = load(evidence/'public-repro-cpu-r1/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution']
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_nonv2_temporal_projection69.py')
    assert {r['stage'] for r in cpu['trials']} == {'decision', 'world'}
    for stage in ('decision', 'world'):
        frozen = family/(stage+'-model-r1')
        identity = load(frozen/'baseline-identity.json')
        assert identity['source_count'] == 59 and identity['installed_baseline'] == 'qualified_original63'
        assert identity['case'] == 'unitree_z1_dual_arm_cleanup_pencils'
        assert identity['official_frame_stride'] == 4 and identity['no_new_collective']
        assert sha(base/identity['manifest_name']) == identity['manifest_SHA256']
        for name, digest in dict(identity['manifest'], **identity['assets']).items():
            assert sha(base/name) == digest, name
        for name, digest in load(frozen/'source-hashes.json').items():
            assert sha(frozen/name) == digest, name
        for name in ('nonv2_ffn59_mode.py', 'nonv2_temporal_block59_mode.py', 'nonv2_all_blocks59_mode.py'):
            assert sha(frozen/'scripts'/name) == sha(prior/name)
        profile = (frozen/'scripts/profile_nccl_geglu_209_pair.py').read_text()
        assert 'baseline_is_original63_nonv2_FP16=True' in profile
        assert "ctl['captured']==66" in profile
        assert "ctl['temporal_output_projection']['captured']==16" in profile
        helper = (frozen/'scripts/nonv2_temporal_projection69_mode.py').read_text()
        assert 'with original_mode(True):' in helper
        assert 'dist.all_gather_into_tensor' in helper
        assert 'parent_inference(value, rank, forward, keep_local, mask)' in helper
        row = next(r for r in cpu['trials'] if r['stage'] == stage)
        assert row['base59_exact'] and row['published_scripts_exact']
        assert row['all_four_CPU_checks_passed'] and row['projection_cases'] == 16
        summary_path = evidence/(stage+'-model-r1')/'summary.json'
        if summary_path.exists():
            summary = load(summary_path)
            assert not summary['official_adopted'] and summary['complete_workload_PSNR_unverified']
            if summary['clean']:
                report = load(summary_path.parent/'nccl-geglu-model-report.json')
                assert report['clean'] and report['passed'] and len(report['ranks']) == 2
                for rank in report['ranks']:
                    assert rank['baseline_is_original63_nonv2_FP16']
                    assert rank['RNG_unchanged'] and rank['baseline_restoration_bitwise']
                    assert rank['zero_FFN_changes_baseline']
                    assert len(rank['cases']) == 6 and len(rank['pairs']) == 6
                    assert all(e['gross_passed'] for c in rank['cases'] for e in c['errors'])
                    controller = rank['FFN_launches']
                    assert controller['captured'] == 66
                    assert controller['temporal_output_projection']['captured'] == 16
                    assert controller['temporal_output_projection']['additional_collectives'] == 0
    outcome = load(evidence/'model-outcome.json')
    for stage in ('decision', 'world'):
        assert outcome[stage]['all_pairs_faster']
        assert all(p['saved_ms'] > 0 for p in outcome[stage]['conservative_pairs'])
        ranks = load(evidence/(stage+'-model-r1')/'nccl-geglu-model-report.json')['ranks']
        assert all(r['partitioned_master_tensors'] == 788 for r in ranks)
        for index, pair in enumerate(outcome[stage]['conservative_pairs']):
            assert pair['fastest_rank_baseline_ms'] == min(r['pairs'][index]['baseline_ms'] for r in ranks)
            assert pair['slowest_rank_candidate_ms'] == max(r['pairs'][index]['candidate_ms'] for r in ranks)
            assert pair['saved_ms'] == pair['fastest_rank_baseline_ms']-pair['slowest_rank_candidate_ms']
    full = family/'full64-r1'
    overlay = full/'production-overlay'
    manifest = load(overlay/'nccl-parallel-production-sources.json')
    previous = repo/'experiments/nonv2-other-blocks63/pencils-full-r1'
    original = load(previous/'production-sources.json')
    runtime = 'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    helper_name = 'src/unifolm_wma/utils/nonv2_temporal_projection69_mode.py'
    assert len(manifest) == 64 and set(manifest)-set(original) == {helper_name}
    assert {n for n,h in original.items() if manifest[n] != h} == {runtime}
    for name, digest in manifest.items():
        p = overlay/name
        assert sha(p if p.exists() else base/name) == digest, name
    for name, digest in load(full/'asset-hashes.json').items():
        assert sha(overlay/name) == digest, name
    for name, digest in load(full/'guard-source-hashes.json').items():
        assert sha(full/'guard'/name) == digest, name
    packaged = (overlay/helper_name).read_text()
    recovered = packaged.replace('from .nonv2_all_blocks59_mode import', 'from nonv2_all_blocks59_mode import')
    recovered = recovered.replace('from . import nonv2_temporal_block59_mode as temporal',
                                  'import nonv2_temporal_block59_mode as temporal')
    assert recovered == (family/'decision-model-r1/scripts/nonv2_temporal_projection69_mode.py').read_text()
    assert load(full/'guard/flags.json') == load(previous/'flags.json')
    assert (overlay/'scripts/evaluation/world_model_interaction.py').read_bytes() == (
        previous/'production-overlay/scripts/evaluation/world_model_interaction.py').read_bytes()
    cpu = load(evidence/'full64-public-repro-cpu-r1/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution'] and cpu['source_count'] == 64
    assert cpu['exact_timed_manifest_and_assets'] and cpu['packaged_projection_CPU_cases'] == 16
    assert cpu['public_script_SHA256'] == sha(repo/'reproduce/run_nonv2_projection64.py')
    for label in ('baseline-full-r2', 'matched-baseline-full-r3', 'full64-r1'):
        if label != 'full64-r1':
            frozen = family/label
            identity = load(frozen/'baseline-identity.json')
            assert identity['manifest'] == original
            assert identity['assets'] == load(previous/'timing-asset-hashes.json')
            for name, digest in load(frozen/'source-hashes.json').items():
                assert sha(frozen/'guard'/name) == digest
        summary = load(evidence/label/'summary.json')
        assert summary['clean'] and summary['workload_complete'] and summary['quality_passed']
        assert summary['rank_videos_exact'] and not summary['official_adopted']
        row = next(r for r in load(evidence/label/'result.json')['cases']
                   if r['case'] == 'unitree_z1_dual_arm_cleanup_pencils')
        assert (row['rounds'], row['videos']) == (8, 17)
        assert row['process_wall_s'] == summary['process_wall_s']
        assert row['psnr_db'] == summary['psnr_db'] and row['psnr_passed']
        complete = load(evidence/label/'parallel/parallel-complete.json')
        assert complete['stage_calls'] == dict(decision=400, world=400)
        for rank in (0, 1):
            r = load(evidence/label/f'parallel/rank{rank}-controller.json')
            assert r['stage_captures'] == dict(decision=33, world=33)
            assert r['controller']['captured'] == 66
            assert all(row['collectives'] == 1 for row in r['controller']['rows'])
            if label == 'full64-r1':
                assert r['controller']['temporal_output_projection']['captured'] == 32
                assert r['controller']['temporal_output_projection']['additional_collectives'] == 0
    comparison = load(evidence/'complete-process-comparison.json')
    assert comparison['baseline_GPUs'] == comparison['candidate_GPUs'] == ['1', '3']
    assert comparison['baseline_s'] == 106.348409 and comparison['candidate_s'] == 105.976138
    assert abs(comparison['saved_s'] - (106.348409-105.976138)) < 1e-12
    assert comparison['single_complete_pair'] and comparison['baseline_formal_videos_exact']
    assert comparison['both_workloads_complete'] and comparison['both_quality_passed']
    assert not comparison['official_adopted']
    print('PROJECTION69_VERIFIED original63_two_GPU_baseline=true six_state_models=true full17_video_quality_passed=true same_pair_saved0_372271s=true not_adopted=true')


if __name__ == '__main__':
    verify()
