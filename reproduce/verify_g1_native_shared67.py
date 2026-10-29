"""Verify G1 temporal-extension evidence; process speed is a separate gate."""
import hashlib,json,statistics
from pathlib import Path


def verify():
    repo=Path(__file__).resolve().parents[1]
    code=repo/'experiments/half-native-shared67'
    evidence=repo/'results/aggressive-20261029/half-native-shared67'
    def load(p):return json.loads(p.read_text())
    for revision in (4,5):
        for stage in ('decision','world'):
            label=f'{stage}-model-r{revision}';frozen=code/label;ev=evidence/label
            for name,digest in load(frozen/'source-hashes.json').items():
                assert hashlib.sha256((frozen/name).read_bytes()).hexdigest()==digest
            identity=load(frozen/'baseline-production-identity.json')
            assert identity['source_count']==59 and identity['official_frame_stride']==6
            assert identity['actual_case']=='unitree_g1_pack_camera'
            flags=load(frozen/'flags.json')
            assert flags==load(repo/'experiments/nonv2-vae-all-display/unitree_g1_pack_camera-flags.json')
            assert flags['UNIFOLM_FP32_FEEDBACK_HEADS']=='1'
            summary=load(ev/'summary.json');assert not summary['official_adopted']
            if revision==4:
                assert not summary['clean'];continue
            report=load(ev/'nccl-geglu-model-report.json')
            assert report['clean'] and report['passed'] and report['ready_for_model']
            assert len(report['ranks'])==2
            for i,rank in enumerate(report['ranks']):
                assert rank==load(ev/f'rank{i}-report.json')
                assert rank['stage']==stage and rank['FFN_master_tensors']==954
                assert rank['RNG_unchanged'] and rank['baseline_restoration_bitwise'] and rank['zero_FFN_changes_baseline']
                assert len(rank['cases'])==6
                assert all(x['signedzero_bits'] and x['relative_l2']==0 and x['max_abs']==0 for case in rank['cases']for x in case['errors'])
                ctl=rank['FFN_launches'];spatial=ctl['spatial_shared_boundary'];temporal=ctl['temporal_shared_boundary']
                assert ctl['captured']==31 and ctl['native_linear_capture_count']==62 and ctl['GEGLU']['captured']==31
                assert spatial['captured']==15 and spatial['attention_calls']==75
                assert temporal['captured']==16 and temporal['attention_calls']==32
                assert spatial['new_collectives']==temporal['new_collectives']==0
                assert sum(x['collectives']for x in ctl['rows'])+spatial['captured']+temporal['captured']==31
                assert temporal['GroupNorm_before_partition_full_input']
                assert all(x['full16_time_tokens']and x['original_constexpr_and_rounding']and len(x['original_CUBIN_SHA256'])==64 for x in temporal['calls'])
                assert len(rank['pairs'])==6 and rank['median_saved_ms']==statistics.median(x['saved_ms']for x in rank['pairs'])
            derived=load(evidence/'model-outcome.json')[stage]
            values=[]
            for i,pair in enumerate(derived['conservative_pairs']):
                b=min(r['pairs'][i]['baseline_ms']for r in report['ranks']);c=max(r['pairs'][i]['candidate_ms']for r in report['ranks'])
                assert pair==dict(fastest_rank_baseline_ms=b,slowest_rank_candidate_ms=c,saved_ms=b-c)
                values.append(b-c)
            assert all(v>0 for v in values)and derived['all_pairs_faster']
            assert derived['conservative_median_saved_ms']==statistics.median(values)
    baseline=load(repo/'experiments/nonv2-vae-all-display/unifolm/nonv2-aggressive-production-sources.json')
    for revision in (1,2):
        frozen=code/f'full-r{revision}';overlay=frozen/'production-overlay';ev=evidence/f'full-r{revision}'
        manifest=load(overlay/'nccl-parallel-production-sources.json')
        assert len(manifest)==67 and all(manifest[n]==h for n,h in baseline.items())
        for path in overlay.glob('src/**/*.py'):
            assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[str(path.relative_to(overlay))]
        for name,digest in load(frozen/'guard-source-hashes.json').items():
            assert hashlib.sha256((frozen/'guard'/name).read_bytes()).hexdigest()==digest
        assets=load(frozen/'asset-hashes.json')
        for name,digest in assets.items():
            if (overlay/name).is_file():assert hashlib.sha256((overlay/name).read_bytes()).hexdigest()==digest
        proof=load(overlay/'nccl-entry-timing-proof.json')
        assert proof['source59_to67'] and proof['four_additive_lifecycle_hooks'] and proof['all_bind_capture_and_extra_replay_inside_process_timer']
        assert proof['temporal_original_CUBIN_and_constexpr_preserved']and proof['full5D_GroupNorm_and16_time_tokens_preserved']
        assert not proof['new_native_argument_inspection']
        runtime=(overlay/'src/unifolm_wma/utils/nccl_nonv2_native_shared_runtime67.py').read_text()
        assert runtime.index('_controller.close_owners()')<runtime.index("entry['graph'].reset()")
        summary=load(ev/'summary.json');assert not summary['official_adopted']
        if revision==1:
            assert not summary['clean'];continue
        assert summary['clean']and summary['workload_complete']and summary['quality_passed']
        assert summary['verified_source_count']==67 and summary['rank_videos_exact']and summary['original_formal_23_video_SHA256_equal']
        result=load(ev/'result.json');case=next(r for r in result['cases']if r['case']=='unitree_g1_pack_camera')
        assert case['rounds']==11 and case['videos']==23 and case['execution_ok']and case['profile_complete']
        assert case['process_wall_s']==summary['process_wall_s']and case['psnr_db']==summary['psnr_db']==26.10715633179594
        parallel=load(ev/'parallel/parallel-complete.json')
        assert parallel['stage_calls']==dict(decision=550,world=550)and parallel['rank_videos_SHA256_equal']
        assert len(parallel['video_hashes'])==23 and sorted(parallel['video_hashes'].values())==sorted(summary['original_formal_video_hashes'].values())
        for rank in (0,1):
            ctl=load(ev/f'parallel/rank{rank}-controller.json')['controller']
            assert ctl['captured']==62 and ctl['native_linear_capture_count']==124 and ctl['GEGLU']['captured']==62
            spatial=ctl['spatial_shared_boundary'];temporal=ctl['temporal_shared_boundary']
            assert spatial['captured']==30 and spatial['attention_calls']==150
            assert temporal['captured']==32 and temporal['attention_calls']==64
            assert sum(x['collectives']for x in ctl['rows'])+spatial['captured']+temporal['captured']==62
    comparison=load(evidence/'complete-comparison.json')
    base=load(evidence/'original59-baseline-r3/summary.json')
    assert base['clean']and base['quality_passed']and base['workload_complete']and base['verified_source_count']==59
    assert comparison['baseline_s']==base['process_wall_s']==158.749762
    assert comparison['candidate_s']==load(evidence/'full-r2/summary.json')['process_wall_s']==158.629774
    assert comparison['saved_s']==round(comparison['baseline_s']-comparison['candidate_s'],6)==0.119988
    assert comparison['one_complete_comparison_only']and not comparison['official_adopted']and comparison['formal_pair_unchanged']
    assert comparison['six_model_pairs_are_separate_from_full_process_comparison']
    assert load(evidence/'original59-baseline-r3/g1-native-half-baseline-sources.json')==baseline
    print('G1 shared67: bitwise models,31 existing gathers,full11/23 unchanged videos;one complete comparison saved0.119988s,not adopted.')


if __name__=='__main__':verify()
