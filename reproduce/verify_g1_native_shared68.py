"""Check exact root-only model transport and complete G1 evidence."""
import hashlib,json
from pathlib import Path


def verify():
    repo=Path(__file__).resolve().parents[1]
    code=repo/'experiments/half-native-shared68';evidence=repo/'results/aggressive-20261029/half-native-shared68'
    def load(path):return json.loads(path.read_text())
    for revision in (1,2,3):
        frozen=code/f'transport-synthetic-r{revision}';ev=evidence/f'transport-synthetic-r{revision}'
        for name,digest in load(frozen/'source-hashes.json').items():assert hashlib.sha256((frozen/name).read_bytes()).hexdigest()==digest
        report=load(ev/'nccl-geglu-model-report.json')
        assert report['clean']and report['passed']and report['synthetic_only']and report['no_official_score']
        assert not report['ready_for_model']
        for i,rank in enumerate(report['ranks']):
            assert rank==load(ev/f'rank{i}-report.json')
            assert rank['synthetic_only']and rank['RNG_unchanged']and rank['all_supported_cases_bitwise']
            assert rank['no_kernel_argument_or_address_exports']and len(rank['cases'])==6
            for case in rank['cases']:
                assert all(e['signedzero_bits']for e in case['errors'])
                row=case['report'];assert row['no_precision_change']
                assert row['fallback_native_cuda']==(case['case']=='noncontiguous_fallback')
                if not row['fallback_native_cuda']:
                    assert row['root_only_H2D']and row['original_parameter_objects_and_aliases']and row['original_contiguous_strides']
                    if revision>=2:assert row['all_tensor_starts_256_byte_aligned']
                    if revision>=3:assert row['standard_cuda_lifecycle_after_broadcast']
    baseline=load(repo/'experiments/nonv2-vae-all-display/unifolm/nonv2-aggressive-production-sources.json')
    original67=load(repo/'experiments/half-native-shared67/full-r2/production-overlay/nccl-parallel-production-sources.json')
    for revision in (1,2,3):
        frozen=code/f'full-r{revision}';overlay=frozen/'production-overlay';ev=evidence/f'full-r{revision}'
        manifest=load(overlay/'nccl-parallel-production-sources.json')
        assert len(manifest)==68 and all(manifest[n]==h for n,h in baseline.items())
        for name,digest in original67.items():
            if name!='src/unifolm_wma/utils/nccl_nonv2_native_shared_runtime67.py':assert manifest[name]==digest
        for path in overlay.glob('src/**/*.py'):assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[str(path.relative_to(overlay))]
        for name,digest in load(frozen/'guard-source-hashes.json').items():assert hashlib.sha256((frozen/'guard'/name).read_bytes()).hexdigest()==digest
        for name,digest in load(frozen/'asset-hashes.json').items():
            if (overlay/name).is_file():assert hashlib.sha256((overlay/name).read_bytes()).hexdigest()==digest
        proof=load(overlay/'nccl-entry-timing-proof.json')
        assert proof['source59_to68']and proof['five_lifecycle_hooks']and not proof['four_additive_lifecycle_hooks']
        assert proof['root_only_H2D_then_typed_CUDA_broadcast']and proof['weight_move_inside_original_model_to_gpu_stage']and proof['no_new_inference_math']
        entry=(overlay/'scripts/evaluation/world_model_interaction.py').read_text()
        assert entry.index('timing.stage("model_to_gpu")')<entry.index('model = _nccl_move_cuda(model)')<entry.index('timing.stage("inference_optimization_setup")')
        summary=load(ev/'summary.json');assert not summary['official_adopted']
        if revision==1:
            assert not summary['clean'];continue
        assert summary['clean']and summary['quality_passed']and summary['workload_complete']and summary['verified_source_count']==68
        assert summary['rank_videos_exact']and summary['original_formal_23_video_SHA256_equal']
        candidate=(overlay/'src/unifolm_wma/utils/nccl_root_model_move.py').read_bytes()
        assert candidate==(code/'transport-synthetic-r3/scripts/nccl_root_model_move.py').read_bytes()
        case=next(r for r in load(ev/'result.json')['cases']if r['case']=='unitree_g1_pack_camera')
        assert case['rounds']==11 and case['videos']==23 and case['execution_ok']and case['profile_complete']
        assert case['process_wall_s']==summary['process_wall_s']and case['psnr_db']==summary['psnr_db']==26.10715633179594
        if revision==2:assert case['process_wall_s']==155.314304
        parallel=load(ev/'parallel/parallel-complete.json')
        assert parallel['stage_calls']==dict(decision=550,world=550)and parallel['rank_videos_SHA256_equal']
        assert len(parallel['video_hashes'])==23 and sorted(parallel['video_hashes'].values())==sorted(summary['original_formal_video_hashes'].values())
        for rank in (0,1):
            move=load(ev/f'parallel/rank{rank}-weight-transport.json')
            assert move['root_only_H2D']and not move['fallback_native_cuda']and move['standard_cuda_lifecycle_after_broadcast']
            assert move['all_tensor_starts_256_byte_aligned']and move['tensors']==3576 and move['buckets']==67 and move['total_bytes']==16718118720
            ctl=load(ev/f'parallel/rank{rank}-controller.json')['controller']
            assert ctl['captured']==62 and ctl['GEGLU']['captured']==62 and ctl['native_linear_capture_count']==124
            spatial=ctl['spatial_shared_boundary'];temporal=ctl['temporal_shared_boundary']
            assert spatial['captured']==30 and spatial['attention_calls']==150 and temporal['captured']==32 and temporal['attention_calls']==64
            assert sum(x['collectives']for x in ctl['rows'])+spatial['captured']+temporal['captured']==62
    outcome=load(evidence/'complete-repeat-outcome.json')
    assert outcome['same_GPU_pair']and outcome['candidate_sources_assets_flags_guard_identical']and outcome['complete_speed_gate_all_pairs_positive']
    assert outcome['separate_from_six_model_pairs']and len(outcome['pairs'])==2
    for row in outcome['pairs']:
        candidate=load(evidence/f"full-r{row['candidate_revision']}/summary.json")
        base_dir=(repo/'results/aggressive-20261029/half-native-shared67/original59-baseline-r3')if row['baseline_revision']==3 else evidence/'original59-baseline-r4'
        original=load(base_dir/'summary.json')
        assert original['clean']and original['quality_passed']and original['workload_complete']
        assert row['candidate_s']==candidate['process_wall_s']and row['baseline_s']==original['process_wall_s']
        assert row['saved_s']==round(row['baseline_s']-row['candidate_s'],6)>0
    assert load(code/'full-r2/production-overlay/nccl-parallel-production-sources.json')==load(code/'full-r3/production-overlay/nccl-parallel-production-sources.json')
    assert load(code/'full-r2/asset-hashes.json')==load(code/'full-r3/asset-hashes.json')
    assert load(code/'full-r2/guard-source-hashes.json')==load(code/'full-r3/guard-source-hashes.json')
    public=load(evidence/'public-repro-cpu-r1/summary.json')
    assert public['passed']and public['no_GPU_execution']and public['source_count']==68 and public['exact_timed_manifest_and_assets']
    assert public['public_script_SHA256']==hashlib.sha256((repo/'reproduce/run_g1_native_shared68.py').read_bytes()).hexdigest()
    accepted=load(code/'accepted-outcome.json');formal=load(repo/'results/result.json');fresh=load(evidence/'official103/validated/result.json')
    assert accepted['official_adopted']and accepted['official103_accepted']and accepted['official_video_count']==103 and accepted['GPUs']==2
    assert accepted['G1_s']==155.314304 and accepted['case_sum_s']==765.731042 and accepted['formal_saved_s']==5.583233
    assert fresh==formal and fresh['status']=='accepted'and fresh['psnr_validation']['video_count']==103 and fresh['psnr_validation']['psnr_ok']
    for name,digest in accepted['pair_hashes'].items():
        assert hashlib.sha256((repo/'results'/name).read_bytes()).hexdigest()==digest
        assert (repo/'results'/name).read_bytes()==(evidence/'official103/validated'/name).read_bytes()
    copies=load(evidence/'official103/validated/immutable-video-copies.json')
    assert copies['count']==103 and copies['ordinary_files']and copies['original_outputs_unchanged']
    print('Adopted G1 shared68: exact transport,standard device lifecycle,two positive complete comparisons,unchanged23 videos,fresh official103 verified.')


if __name__=='__main__':verify()
