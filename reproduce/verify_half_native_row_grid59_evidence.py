"""Verify exact original-kernel G1 FFN screening, without claiming video quality."""
import hashlib,json,statistics
from pathlib import Path

def verify():
    root=Path(__file__).resolve().parents[1]
    code=root/'experiments/half-native-row-grid59'
    evidence=root/'results/aggressive-20261028/half-native-row-grid59'
    def load(path):return json.loads(path.read_text())
    for folder in code.iterdir():
        if not (folder/'source-hashes.json').is_file():continue
        for name,digest in load(folder/'source-hashes.json').items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
    synthetic=load(evidence/'synthetic-r1/nccl-geglu-model-report.json')
    assert synthetic['clean'] and synthetic['passed'] and synthetic['synthetic_only'] and synthetic['no_official_score']
    for rank in synthetic['ranks']:
        assert rank['supported_shapes']==9 and rank['all_supported_cases_bitwise']
        assert rank['no_kernel_argument_or_address_exports'] and len(rank['rows'])==10
        for row in rank['rows']:
            if not row['supported']:
                assert row['shape']==[640,10240,1280]
                continue
            assert len(row['cases'])==6
            assert all(x['full_reconstruction_signedzero_bits'] and x['owned_rows_and_unowned_rows_and_padding_passed'] and x['owned_addmm_and_F_linear_bitwise'] for x in row['cases'])
            assert row['metadata']['only_owned_padded_Torch_views_bound'] and row['metadata']['no_argument_bytes_addresses_or_offsets_exported']
    outcome=load(evidence/'model-outcome.json')
    for stage in ('decision','world'):
        for revision in (1,2,3):
            label=f'g1-{stage}-model-r{revision}'
            identity=load(code/label/'baseline-production-identity.json')
            assert identity['source_count']==59 and identity['actual_case']=='unitree_g1_pack_camera' and identity['official_frame_stride']==6
            flags=load(code/label/'flags.json')
            assert flags==load(root/'experiments/nonv2-vae-all-display/unitree_g1_pack_camera-flags.json') and flags['UNIFOLM_FP32_FEEDBACK_HEADS']=='1'
            report=load(evidence/label/'nccl-geglu-model-report.json')
            assert report['clean'] and report['passed'] and report['ready_for_model'] and report['no_official_score']
            for index,rank in enumerate(report['ranks']):
                assert rank==load(evidence/label/f'rank{index}-report.json')
                assert rank['stage']==stage and rank['FFN_master_tensors']==(580 if revision==3 else 132)
                assert rank['RNG_unchanged'] and rank['baseline_restoration_bitwise'] and rank['zero_FFN_changes_baseline']
                assert len(rank['cases'])==6 and all(x['signedzero_bits'] and x['relative_l2']==0 and x['max_abs']==0 for case in rank['cases'] for x in case['errors'])
                assert len(rank['pairs'])==6 and all(x['saved_ms']>0 for x in rank['pairs'])
                assert rank['median_saved_ms']==statistics.median(x['saved_ms'] for x in rank['pairs'])
                ctl=rank['FFN_launches']
                assert ctl['captured']==31 and ctl['native_linear_capture_count']==62 and ctl['new_JIT_kernels']==0
                assert len(ctl['rows'])==31 and ctl['additional_collectives_per_forward']==31
                binding,=ctl['native_binding'];assert len(binding['kernels'])==62 and binding['outputs_poisoned_before_first_partitioned_replay']
                assert all(x['exact_owned_Torch_input_output_weight_binding'] and x['original_compiled_function_and_K_order'] and x['no_argument_bytes_addresses_or_offsets_exported'] for x in binding['kernels'])
                if revision>=2:
                    assert ctl['GEGLU']['captured']==31 and ctl['GEGLU']['original_compiled_CUBIN_matched']
                    assert all(len(x['original_CUBIN_SHA256'])==64 and x['original_N_constexpr_and_rounding'] and x['owned_half_tiles_without_tail_mask_changes'] for x in ctl['GEGLU']['rows'])
                if revision==3:
                    spatial=ctl['spatial_shared_boundary']
                    assert spatial['captured']==15 and spatial['attention_calls']==75 and spatial['new_collectives']==0
                    assert sum(x['collectives'] for x in ctl['rows'])+spatial['captured']==31
                    assert all(x['original_global_mask_slice'] and x['local_batch']==8 and x['no_new_collective'] for x in spatial['calls'])
                    assert all(min(r['pairs'][i]['baseline_ms'] for r in report['ranks'])>max(r['pairs'][i]['candidate_ms'] for r in report['ranks']) for i in range(6))
            if revision==2:
                derived=outcome[stage]
                assert derived['rank_median_saved_ms']==[x['median_saved_ms'] for x in report['ranks']]
                for i,pair in enumerate(derived['conservative_pairs']):
                    b=min(x['pairs'][i]['baseline_ms'] for x in report['ranks']);c=max(x['pairs'][i]['candidate_ms'] for x in report['ranks'])
                    assert pair==dict(fastest_rank_baseline_ms=b,slowest_rank_candidate_ms=c,saved_ms=b-c) and b>c
                assert derived['conservative_median_saved_ms']==statistics.median(x['saved_ms'] for x in derived['conservative_pairs'])
    assert outcome['no_official_score'] and outcome['full_video_PSNR_unverified']
    full=code/'full64-r1';overlay=full/'production-overlay';ev=evidence/'full64-r1'
    manifest=load(overlay/'nccl-parallel-production-sources.json')
    baseline=load(root/'experiments/nonv2-vae-all-display/unifolm/nonv2-aggressive-production-sources.json')
    assert len(manifest)==64 and len(baseline)==59 and all(manifest[n]==h for n,h in baseline.items())
    for path in overlay.glob('src/**/*.py'):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[str(path.relative_to(overlay))]
    proof=load(overlay/'nccl-entry-timing-proof.json')
    assert proof['source59_to64'] and proof['four_additive_lifecycle_hooks'] and proof['all_bind_capture_and_extra_replay_inside_process_timer']
    assert proof['two_graph_ownership_and_restore_before_reset'] and proof['qualified_FFNs_per_stage']==31
    runtime=(overlay/'src/unifolm_wma/utils/nccl_nonv2_native_ffn_runtime64.py').read_text()
    assert runtime.index('_controller.close_owners()')<runtime.index("entry['graph'].reset()")
    summary=load(ev/'summary.json');assert summary['clean'] and summary['workload_complete'] and summary['quality_passed']
    assert summary['verified_source_count']==64 and summary['rounds']==11 and summary['videos']==23
    assert summary['original_formal_23_video_SHA256_equal'] and not summary['official_adopted']
    result=load(ev/'result.json');case=next(x for x in result['cases'] if x['case']=='unitree_g1_pack_camera')
    assert case['process_wall_s']==summary['process_wall_s']==159.604883
    assert case['psnr_db']==summary['psnr_db']==26.10715633179594 and case['psnr_passed']
    assert case['rounds']==11 and case['videos']==23 and case['execution_ok'] and case['profile_complete']
    parallel=load(ev/'parallel/parallel-complete.json')
    assert parallel['stage_captures']==dict(decision=31,world=31) and parallel['stage_calls']==dict(decision=550,world=550)
    assert parallel['rank_videos_SHA256_equal'] and len(parallel['video_hashes'])==23
    assert sorted(parallel['video_hashes'].values())==sorted(summary['original_formal_video_hashes'].values())
    for rank in (0,1):
        ctl=load(ev/f'parallel/rank{rank}-controller.json')['controller']
        assert ctl['captured']==62 and ctl['native_linear_capture_count']==124 and ctl['GEGLU']['captured']==62
        assert len(ctl['native_binding'])==2 and all(len(x['kernels'])==62 for x in ctl['native_binding'])
    baseline=load(evidence/'original59-baseline-r1/summary.json')
    assert baseline['clean'] and baseline['workload_complete'] and baseline['quality_passed'] and baseline['verified_source_count']==59
    assert baseline['process_wall_s']==157.6129 and baseline['psnr_db']==summary['psnr_db']
    assert summary['same_GPU_original59_baseline_s']==baseline['process_wall_s']
    assert summary['same_GPU_complete_saved_s']==round(baseline['process_wall_s']-summary['process_wall_s'],6)==-1.991983
    assert not summary['full_process_speed_passed'] and summary['rejected_for_adoption'] and not summary['official103_pending']
    shared=evidence/'full66-r1';summary=load(shared/'summary.json')
    assert summary['clean'] and summary['workload_complete'] and summary['quality_passed'] and summary['verified_source_count']==66
    assert summary['process_wall_s']==157.103843 and summary['psnr_db']==26.10715633179594
    assert not summary['official_adopted'] and summary['adoption_pending_repeat_or_setup_optimization']
    assert summary['same_GPU_complete_saved_s']==round(157.6129-summary['process_wall_s'],6)==0.509057
    parallel=load(shared/'parallel/parallel-complete.json')
    assert parallel['stage_calls']==dict(decision=550,world=550) and parallel['rank_videos_SHA256_equal']
    assert len(parallel['video_hashes'])==23 and sorted(parallel['video_hashes'].values())==sorted(summary['original_formal_video_hashes'].values())
    for rank in (0,1):
        ctl=load(shared/f'parallel/rank{rank}-controller.json')['controller'];scope=ctl['spatial_shared_boundary']
        assert ctl['captured']==62 and ctl['GEGLU']['captured']==62 and ctl['native_linear_capture_count']==124
        assert scope['captured']==30 and scope['attention_calls']==150 and scope['new_collectives']==0
        assert sum(x['collectives'] for x in ctl['rows'])+scope['captured']==62
    public=load(evidence/'public-repro-cpu-r1/summary.json')
    assert public['passed'] and public['no_GPU_execution'] and public['source_count']==66 and public['exact_timed_manifest_and_assets']
    assert public['public_script_SHA256']==hashlib.sha256((root/'reproduce/run_g1_native_shared66.py').read_bytes()).hexdigest()
    print('Original Half FFN grids:G1 bitwise model/full23-video checks passed;fresh full baseline157.612900 beats candidate159.604883;not adopted.')

if __name__=='__main__':verify()
