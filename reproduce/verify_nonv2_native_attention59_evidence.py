"""Verify the quality-screened but speed-rejected G1 attention experiment."""
import hashlib,json,statistics
from pathlib import Path


def verify():
    root=Path(__file__).resolve().parents[1]
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    code=root/'experiments/nonv2-native-attention59';ev=root/'results/aggressive-20261028/nonv2-native-attention59'
    base=load(root/'experiments/nonv2-vae-all-display/unifolm/nonv2-aggressive-production-sources.json');assert len(base)==59
    outcome=load(ev/'decision-world-outcome.json')
    assert outcome['full_case_not_run']and not outcome['official_adopted']and outcome['official_pair_unchanged']and outcome['not_an_official_score_or_process_time']
    for stage in('decision','world'):
        for revision in(1,2):
            folder=code/f'g1-{stage}-r{revision}';evidence=ev/f'g1-{stage}-r{revision}'
            hashes=load(folder/'source-hashes.json')
            for n,h in hashes.items():assert sha(folder/n)==h,n
            identity=load(folder/'baseline-production-identity.json');assert identity['source_count']==59 and identity['manifest']==base
            assert identity['actual_case']=='unitree_g1_pack_camera'and identity['official_frame_stride']==6 and identity['G1_FP32_feedback_heads_preserved']
            flags=load(folder/'flags.json');assert flags==load(root/'experiments/nonv2-vae-all-display/unitree_g1_pack_camera-flags.json')and flags['UNIFOLM_FP32_FEEDBACK_HEADS']=='1'
            assert 'NONV2_NATIVE_ATTENTION59_CPU passed=true'in(folder/'cpu-preflight.log').read_text()
            rank_source=(folder/'check_nccl_geglu_model_rank.py').read_text()
            assert "'--dataset','unitree_g1_pack_camera'"in rank_source
            assert f"'--frame_stride','{4 if revision==1 else 6}'"in rank_source
            assert identity['diagnostic_frame_stride']==(4 if revision==1 else 6)
            summary=load(evidence/'summary.json');assert summary['full_video_PSNR_unverified']and summary['no_official_score']and not summary['official_adopted']
            if revision==1:
                assert not summary['clean']and summary['failed_before_denoiser_input']
                assert 'Required input cache does not exist:'in(evidence/'attempt-1/ranks.log').read_text()
                continue
            assert summary['clean']and summary['ready_for_full_trial']is False
            report=load(evidence/'nccl-geglu-model-report.json');assert report['clean']and report['passed']and report['no_official_score']and report['ready_for_model']is False
            ranks=report['ranks'];assert len(ranks)==2
            for index,row in enumerate(ranks):
                assert row==load(evidence/f'rank{index}-report.json')
                assert row['stage']==stage and row['baseline_is_original59_nonv2_FP16']and row['G1_FP32_feedback_head_policy_preserved']
                assert row['partitioned_master_tensors']==512 and row['RNG_unchanged']and row['baseline_restoration_bitwise']and row['zero_partitioned_masters_changes_baseline']
                assert len(row['cases'])==6 and all(x['signedzero_bits']and x['gross_passed']for case in row['cases']for x in case['errors'])
                assert len(row['pairs'])==6 and row['median_saved_ms']==statistics.median(x['saved_ms']for x in row['pairs'])
                ctl=row['attention_launches'];assert ctl['captured']==16 and ctl['native_attention_calls']==80 and ctl['additional_collectives_per_forward']==16
                assert len(ctl['rows'])==16 and all(x['collectives']==1 and x['original_full16_GEMM_FFN_projection_descriptors']for x in ctl['rows'])
                assert len(ctl['calls'])==80 and all(x['local_batch']==8 and x['outer_batch']==16 and x['global_mask_slice']and x['extra_collectives']==0 for x in ctl['calls'])
            derived=outcome[stage];assert derived['all_six_states_three_outputs_bitwise']and not derived['six_pair_conservative_gate']and not derived['raw_all_rank_all_pair_speed_gate']
            assert derived['raw_rank_median_saved_ms']==[r['median_saved_ms']for r in ranks]
            for index,pair in enumerate(derived['pairs']):
                raw=[r['pairs'][index]for r in ranks]
                fast=min(x['baseline_ms']for x in raw);candidate=max(x['candidate_ms']for x in raw)
                assert pair['fastest_rank_baseline_ms']==fast and pair['slowest_rank_candidate_ms']==candidate
                assert pair['fastest_single_rank_minus_candidate_ms']==fast-candidate<0
            assert derived['median_fastest_single_rank_minus_candidate_ms']==statistics.median(x['fastest_single_rank_minus_candidate_ms']for x in derived['pairs'])
    current=load(root/'results/result.json');g1=next(x for x in current['cases']if x['case']=='unitree_g1_pack_camera')
    assert g1['process_wall_s']==160.897537 and g1['psnr_db']==26.10715633179594
    print('G1 attention-only:512 masters/six-state bitwise passed;two-stage paired speed gates failed;full-video qualification not claimed;formal G1 unchanged.')


if __name__=='__main__':verify()
