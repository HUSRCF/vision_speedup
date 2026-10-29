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
        for revision in (1,2):
            label=f'g1-{stage}-model-r{revision}'
            identity=load(code/label/'baseline-production-identity.json')
            assert identity['source_count']==59 and identity['actual_case']=='unitree_g1_pack_camera' and identity['official_frame_stride']==6
            flags=load(code/label/'flags.json')
            assert flags==load(root/'experiments/nonv2-vae-all-display/unitree_g1_pack_camera-flags.json') and flags['UNIFOLM_FP32_FEEDBACK_HEADS']=='1'
            report=load(evidence/label/'nccl-geglu-model-report.json')
            assert report['clean'] and report['passed'] and report['ready_for_model'] and report['no_official_score']
            for index,rank in enumerate(report['ranks']):
                assert rank==load(evidence/label/f'rank{index}-report.json')
                assert rank['stage']==stage and rank['FFN_master_tensors']==132
                assert rank['RNG_unchanged'] and rank['baseline_restoration_bitwise'] and rank['zero_FFN_changes_baseline']
                assert len(rank['cases'])==6 and all(x['signedzero_bits'] and x['relative_l2']==0 and x['max_abs']==0 for case in rank['cases'] for x in case['errors'])
                assert len(rank['pairs'])==6 and all(x['saved_ms']>0 for x in rank['pairs'])
                assert rank['median_saved_ms']==statistics.median(x['saved_ms'] for x in rank['pairs'])
                ctl=rank['FFN_launches']
                assert ctl['captured']==31 and ctl['native_linear_capture_count']==62 and ctl['new_JIT_kernels']==0
                assert len(ctl['rows'])==31 and ctl['additional_collectives_per_forward']==31
                binding,=ctl['native_binding'];assert len(binding['kernels'])==62 and binding['outputs_poisoned_before_first_partitioned_replay']
                assert all(x['exact_owned_Torch_input_output_weight_binding'] and x['original_compiled_function_and_K_order'] and x['no_argument_bytes_addresses_or_offsets_exported'] for x in binding['kernels'])
                if revision==2:
                    assert ctl['GEGLU']['captured']==31 and ctl['GEGLU']['original_compiled_CUBIN_matched']
                    assert all(len(x['original_CUBIN_SHA256'])==64 and x['original_N_constexpr_and_rounding'] and x['owned_half_tiles_without_tail_mask_changes'] for x in ctl['GEGLU']['rows'])
            if revision==2:
                derived=outcome[stage]
                assert derived['rank_median_saved_ms']==[x['median_saved_ms'] for x in report['ranks']]
                for i,pair in enumerate(derived['conservative_pairs']):
                    b=min(x['pairs'][i]['baseline_ms'] for x in report['ranks']);c=max(x['pairs'][i]['candidate_ms'] for x in report['ranks'])
                    assert pair==dict(fastest_rank_baseline_ms=b,slowest_rank_candidate_ms=c,saved_ms=b-c) and b>c
                assert derived['conservative_median_saved_ms']==statistics.median(x['saved_ms'] for x in derived['conservative_pairs'])
    assert outcome['no_official_score'] and outcome['full_video_PSNR_unverified']
    print('Original Half FFN grids:9/10 shapes qualified;G1 two-stage six-state bitwise and six paired speed checks passed;full-video quality not inferred.')

if __name__=='__main__':verify()
