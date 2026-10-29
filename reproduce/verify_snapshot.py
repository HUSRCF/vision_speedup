"""Verify fixed production source and historical official-result evidence."""
import hashlib,json,re
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    root=Path(__file__).resolve().parents[1];code=root/'unifolm'
    manifest=json.loads((code/'world-head-filter-cache-production-sources.json').read_text())
    assert len(manifest)==209
    assert set(manifest)=={str(p.relative_to(code)) for p in (code/'src').rglob('*.py')}
    for name,digest in manifest.items():assert sha(code/name)==digest,name
    assert sha(code/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so')=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    assert sha(root/'results/runtime.log')=='cdc13a339e27e45067d0fa4679c625a81d501fe5fa37ee5a30e0d8ae1c8f83d1'
    assert sha(root/'results/result.json')=='0eabe70d78ebe28361430356ce30f49b58ea99416d50a720406029907d9bbb10'
    history=root/'results/history/official-before-g1-native68'
    assert sha(history/'runtime.log')=='ceeead88c2b68307c155793c17cb249b8d5333e9a3231c1aaac5a999a1b36e54'
    assert sha(history/'result.json')=='464f499a003c971b0fac3e3497310ea339ae124f705de94ddabd0ca813682e2b'
    result=json.loads((root/'results/result.json').read_text());assert result['status']=='accepted'
    cases=result['cases'];assert len(cases)==5 and sum(c['videos'] for c in cases)==103
    assert [c['rounds'] for c in cases]==[12,7,11,8,11]
    assert abs(sum(c['process_wall_s'] for c in cases)-738.805454)<1e-6
    log=(root/'results/runtime.log').read_text()
    timings=re.findall(r'^TIMING case=(\S+) process_wall_s=([0-9.]+)$',log,re.M)
    assert len(timings)==5 and len({case for case,_ in timings})==5
    assert {c['case']:c['process_wall_s'] for c in cases}=={case:float(value) for case,value in timings}
    assert all(c['psnr_passed'] and c['execution_ok'] and c['profile_complete'] for c in cases)
    nonv2=root/'experiments/nonv2-vae-all-display'
    variant=nonv2/'unifolm';other=json.loads((variant/'nonv2-aggressive-production-sources.json').read_text())
    assert len(other)==59
    for name,digest in other.items():assert sha(variant/name)==digest,name
    for name,digest in json.loads((nonv2/'asset-sha256.json').read_text()).items():assert sha(variant/name)==digest,name
    accepted=json.loads((nonv2/'accepted-cases.json').read_text())
    boundary_root=root/'experiments/initial-boundary-221/world-model-r3'
    for name,digest in json.loads((boundary_root/'source-hashes.json').read_text()).items():assert sha(boundary_root/name)==digest,name
    boundary_evidence=json.loads((root/'results/aggressive-20261028/initial-boundary-221/world-model-r3/summary.json').read_text())
    assert boundary_evidence['clean'] and boundary_evidence['passed'] and boundary_evidence['ready_for_model']
    assert boundary_evidence['baseline_source_count']==221 and not boundary_evidence['full_V2_process_or_video_quality_qualification']
    for rank in boundary_evidence['ranks']:
        assert len(rank['cases'])==6 and rank['RNG_unchanged'] and rank['baseline_restoration_bitwise']
        assert all(e['signedzero_bits'] for case in rank['cases'] for e in case['errors'])
        assert len(rank['pairs'])==6 and all(pair['saved_ms']>0 for pair in rank['pairs'])
        ctl=rank['attention_launches'];assert ctl['captured']==1 and ctl['full5D_norm_and16_time_steps_preserved']
        assert ctl['rows'][0]['grouped_copy_nodes']==2 and ctl['rows'][0]['additional_collectives']==0
        assert ctl['rows'][0]['existing_gather_moved_after_residual']
    for stage in ('decision','world'):
        trial=root/f'experiments/nonv2-ffn-59/{stage}-model-r1'
        for name,digest in json.loads((trial/'source-hashes.json').read_text()).items():assert sha(trial/name)==digest,name
        ev=json.loads((root/f'results/aggressive-20261028/nonv2-ffn-59/{stage}-model-r1/summary.json').read_text())
        assert ev['passed'] and ev['clean'] and ev['ready_for_model']
        assert ev['baseline_source_count']==59 and not ev['full_video_quality_qualified']
        for rank in ev['ranks']:
            assert rank['baseline_is_original59_nonv2_FP16'] and rank['FFN_master_tensors']==132
            assert rank['RNG_unchanged'] and rank['baseline_restoration_bitwise'] and rank['zero_FFN_changes_baseline']
            assert rank['full_video_PSNR_unverified'] and rank['FFN_launches']['captured']==33
            assert len(rank['cases'])==6 and all(e['gross_passed'] for case in rank['cases'] for e in case['errors'])
            assert len(rank['pairs'])==6 and all(pair['saved_ms']>0 for pair in rank['pairs'])
    assert sha(root/'experiments/nonv2-ffn-59/decision-model-r1/scripts/nonv2_ffn59_mode.py')==sha(root/'experiments/nonv2-ffn-59/world-model-r1/scripts/nonv2_ffn59_mode.py')
    other63=json.loads((root/'experiments/nonv2-other-blocks63/accepted-outcome.json').read_text())
    other_prediction=json.loads((root/'experiments/nonv2-other-prediction65/accepted-outcome.json').read_text())
    late23=json.loads((root/'experiments/nonv2-late23-prediction65/accepted-outcome.json').read_text())
    dual_late23=json.loads((root/'experiments/nonv2-late23-prediction65/dual-accepted-outcome.json').read_text())
    for row in cases:
        if row['case'] in dual_late23['cases']:
            selected=dual_late23['cases'][row['case']]
            assert row['process_wall_s']==selected['current_s']and row['psnr_db']==selected['psnr_db']
        elif row['case'] in late23['cases']:
            selected=late23['cases'][row['case']]
            assert row['process_wall_s']==selected['current_s']and row['psnr_db']==selected['psnr_db']
        elif row['case'] in other_prediction['cases']:
            selected=other_prediction['cases'][row['case']]
            assert row['process_wall_s']==selected['current_s']and row['psnr_db']==selected['psnr_db']
        elif row['case'] in other63['cases']:
            selected=other63['cases'][row['case']]
            assert row['process_wall_s']==selected['current_s']and row['psnr_db']==selected['psnr_db']
        elif row['case']=='unitree_z1_stackbox':
            blocks=json.loads((root/'experiments/nonv2-prediction65/accepted-outcome.json').read_text())
            assert blocks['official_adopted']and row['process_wall_s']==blocks['Z1_s']and row['psnr_db']==blocks['PSNR_dB']
        elif row['case']=='unitree_g1_pack_camera':
            current_g1=json.loads((root/'experiments/half-native-shared68/accepted-outcome.json').read_text())
            assert current_g1['official_adopted']and current_g1['official103_accepted']and current_g1['source_count']==68
            assert row['process_wall_s']==current_g1['G1_s']and row['psnr_db']==current_g1['PSNR_dB']
        elif row['case'] in accepted:assert row['process_wall_s']==accepted[row['case']]['process_wall_s']
    import importlib.util
    overlay=root/'experiments/nccl-ffn-209/timed-entry'
    spec=importlib.util.spec_from_file_location('parallel_sources',overlay/'nccl_parallel209_sources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files,proof=module.production_sources(code,'world-ffn-temporal')
    assert proof==json.loads((overlay/'timing-proof.json').read_text())
    parallel=json.loads((overlay/'production-sources.json').read_text());assert len(parallel)==213
    for name,digest in parallel.items():
        actual=hashlib.sha256(files[name].encode()).hexdigest() if name in files else sha(code/name)
        assert actual==digest,name
    assets=json.loads((overlay/'production-assets.json').read_text())
    assert hashlib.sha256(files['scripts/evaluation/world_model_interaction.py'].encode()).hexdigest()==assets['scripts/evaluation/world_model_interaction.py']
    assert sha(overlay/'timing-proof.json')==assets['nccl-entry-timing-proof.json']
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert sha(code/name)==assets[name]
    outcome=json.loads((overlay/'accepted-outcome.json').read_text());assert outcome['GPUs']==2
    assert outcome['V2_s']==312.462900 # historical213-source result
    import sys
    latest=root/'experiments/nccl-spatial-temporal-209'
    sys.path.insert(0,str(latest))
    try:
        from nccl_spatial_parallel209_sources import production_sources as spatial_sources
        new_files,new_proof=spatial_sources(code,'world-spatial-temporal')
    finally:sys.path.pop(0)
    assert new_proof==json.loads((latest/'timing-proof.json').read_text())
    new_manifest=json.loads((latest/'production-sources.json').read_text());assert len(new_manifest)==215
    for name,digest in new_manifest.items():
        actual=hashlib.sha256(new_files[name].encode()).hexdigest() if name in new_files else sha(code/name)
        assert actual==digest,name
    assets=json.loads((latest/'production-assets.json').read_text())
    assert hashlib.sha256(new_files['scripts/evaluation/world_model_interaction.py'].encode()).hexdigest()==assets['scripts/evaluation/world_model_interaction.py']
    assert sha(latest/'timing-proof.json')==assets['nccl-entry-timing-proof.json']
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert sha(code/name)==assets[name]
    accepted=json.loads((latest/'accepted-outcome.json').read_text())
    assert accepted['GPUs']==2 and accepted['source_count']==215 and accepted['all103_officially_validated']
    assert accepted['V2_s']==301.930663 # historical world-only spatial215 overlay
    latest=root/'experiments/nccl-both-spatial-temporal-209'
    spec=importlib.util.spec_from_file_location('both_parallel_frozen',latest/'production_sources.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    files,proof=frozen.production_sources(code)
    assert proof==json.loads((latest/'timing-proof.json').read_text())
    manifest=json.loads((latest/'production-sources.json').read_text());assert len(manifest)==215
    for name,digest in manifest.items():
        actual=hashlib.sha256(files[name].encode()).hexdigest() if name in files else sha(code/name)
        assert actual==digest,name
    assets=json.loads((latest/'production-assets.json').read_text())
    assert hashlib.sha256(files['scripts/evaluation/world_model_interaction.py'].encode()).hexdigest()==assets['scripts/evaluation/world_model_interaction.py']
    assert sha(latest/'timing-proof.json')==assets['nccl-entry-timing-proof.json']
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert sha(code/name)==assets[name]
    accepted=json.loads((latest/'accepted-outcome.json').read_text())
    assert accepted['GPUs']==2 and accepted['source_count']==215 and accepted['all103_officially_validated']
    assert accepted['V2_s']==289.486308 # historical both-stage215 overlay
    latest=root/'experiments/nccl-native-frame-grid-209'
    spec=importlib.util.spec_from_file_location('native_grid_parallel_frozen',latest/'production_sources.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    files,proof=frozen.production_sources(code)
    assert proof==json.loads((latest/'timing-proof.json').read_text())
    manifest=json.loads((latest/'production-sources.json').read_text());assert len(manifest)==218
    for name,digest in manifest.items():
        actual=hashlib.sha256(files[name].encode()).hexdigest() if name in files else sha(code/name)
        assert actual==digest,name
    assets=json.loads((latest/'production-assets.json').read_text())
    assert hashlib.sha256(files['scripts/evaluation/world_model_interaction.py'].encode()).hexdigest()==assets['scripts/evaluation/world_model_interaction.py']
    assert sha(latest/'timing-proof.json')==assets['nccl-entry-timing-proof.json']
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert sha(code/name)==assets[name]
    accepted=json.loads((latest/'accepted-outcome.json').read_text())
    assert accepted['GPUs']==2 and accepted['source_count']==218 and accepted['all103_officially_validated']
    assert accepted['V2_s']==287.745486 # historical218-source result
    latest=root/'experiments/native-attention-218/production219'
    spec=importlib.util.spec_from_file_location('native_attention219_frozen',latest/'production_sources.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    files,proof=frozen.production_sources(code)
    assert proof==json.loads((latest/'timing-proof.json').read_text())
    manifest=json.loads((latest/'production-sources.json').read_text());assert len(manifest)==219
    for name,digest in manifest.items():
        actual=hashlib.sha256(files[name].encode()).hexdigest() if name in files else sha(code/name)
        assert actual==digest,name
    for name,digest in json.loads((latest/'guard-source-hashes.json').read_text()).items():assert sha(latest/name)==digest,name
    assets=json.loads((latest/'production-assets.json').read_text())
    assert hashlib.sha256(files['scripts/evaluation/world_model_interaction.py'].encode()).hexdigest()==assets['scripts/evaluation/world_model_interaction.py']
    assert sha(latest/'timing-proof.json')==assets['nccl-entry-timing-proof.json']
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert sha(code/name)==assets[name]
    accepted=json.loads((latest/'accepted-outcome.json').read_text())
    assert accepted['GPUs']==2 and accepted['source_count']==219 and accepted['all103_officially_validated']
    assert accepted['V2_s']==280.850431 and accepted['saved_from_matched_baseline_s']==5.874246
    latest=root/'experiments/initial-ffn-219/production220'
    spec=importlib.util.spec_from_file_location('initial_ffn220_frozen',latest/'production_sources.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    files,proof=frozen.production_sources(code)
    assert proof==json.loads((latest/'timing-proof.json').read_text())
    manifest=json.loads((latest/'production-sources.json').read_text());assert len(manifest)==220
    for name,digest in manifest.items():
        actual=hashlib.sha256(files[name].encode()).hexdigest() if name in files else sha(code/name)
        assert actual==digest,name
    for name,digest in json.loads((latest/'guard-source-hashes.json').read_text()).items():assert sha(latest/name)==digest,name
    assets=json.loads((latest/'production-assets.json').read_text())
    assert hashlib.sha256(files['scripts/evaluation/world_model_interaction.py'].encode()).hexdigest()==assets['scripts/evaluation/world_model_interaction.py']
    assert sha(latest/'timing-proof.json')==assets['nccl-entry-timing-proof.json']
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert sha(code/name)==assets[name]
    accepted=json.loads((latest/'accepted-outcome.json').read_text())
    assert accepted['GPUs']==2 and accepted['source_count']==220 and accepted['all103_officially_validated']
    assert accepted['V2_s']==277.436969 and accepted['saved_from_matched_baseline_s']==1.459912
    latest=root/'experiments/temporal-block-220/production221'
    spec=importlib.util.spec_from_file_location('temporal221_pending_frozen',latest/'production_sources.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    files,proof=frozen.production_sources(code)
    assert proof==json.loads((latest/'timing-proof.json').read_text())
    manifest221=json.loads((latest/'production-sources.json').read_text());assert len(manifest221)==221
    for name,digest in manifest221.items():
        actual=hashlib.sha256(files[name].encode()).hexdigest() if name in files else sha(code/name)
        assert actual==digest,name
    accepted221=json.loads((latest/'accepted-outcome.json').read_text())
    assert accepted221['official_adopted'] and accepted221['all103_officially_validated'] and accepted221['source_count']==221
    assert accepted221['V2_s']==270.825199 and accepted221['saved_from_historical220_s']==6.611770
    assert not accepted221['contemporaneous_whole_process_AB_pair']
    assert not accepted221['original_split_root_video_validator_passed'] and accepted221['ordinary_packaging_23_video_validation_passed']
    assert next(c['process_wall_s'] for c in json.loads((root/'results/history/official221-z1-fused-before222/result.json').read_text())['cases']if c['case'].endswith('stackbox_v2'))==accepted221['V2_s']
    split=root/'experiments/temporal-block-220/full-split-r3'
    for name,digest in json.loads((split/'source-hashes.json').read_text()).items():assert sha(split/name)==digest,name
    original209=json.loads((code/'world-head-filter-cache-production-sources.json').read_text())
    assert all(manifest221[n]==h for n,h in original209.items())
    for name,digest in json.loads((latest/'guard-source-hashes.json').read_text()).items():assert sha(latest/name)==digest,name
    assets=json.loads((latest/'production-assets.json').read_text())
    assert hashlib.sha256(files['scripts/evaluation/world_model_interaction.py'].encode()).hexdigest()==assets['scripts/evaluation/world_model_interaction.py']
    assert sha(latest/'timing-proof.json')==assets['nccl-entry-timing-proof.json']
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert sha(code/name)==assets[name]
    evidence=json.loads((root/'results/aggressive-20261028/temporal-block-220/model-r2/summary.json').read_text())
    assert evidence['passed'] and evidence['clean'] and evidence['ready_for_model']
    assert not evidence['runner_archive_copy_completed']
    for rank in evidence['ranks']:
        assert len(rank['cases'])==6 and rank['RNG_unchanged']
        assert all(e['signedzero_bits'] for case in rank['cases'] for e in case['errors'])
        assert len(rank['pairs'])==6 and all(pair['saved_ms']>0 for pair in rank['pairs'])
        ctl=rank['attention_launches'];assert ctl['captured']==16 and ctl['no_new_collectives']
        assert sum(row['grouped_copy_nodes'] for row in ctl['rows'])==54
    pending=root/'experiments/initial-boundary-221/production222'
    spec=importlib.util.spec_from_file_location('initial222_pending_frozen',pending/'production_sources.py')
    frozen222=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen222)
    files222,proof222=frozen222.production_sources(code)
    manifest222=json.loads((pending/'production-sources.json').read_text());assert len(manifest222)==222
    for name,digest in manifest222.items():
        actual=hashlib.sha256(files222[name].encode()).hexdigest() if name in files222 else sha(code/name)
        assert actual==digest,name
    for name,digest in json.loads((pending/'guard-source-hashes.json').read_text()).items():assert sha(pending/name)==digest,name
    assert proof222==json.loads((pending/'timing-proof.json').read_text())
    failed=json.loads((root/'results/aggressive-20261028/initial-boundary-221/full-storage-failed-r1/summary.json').read_text())
    assert failed['status']=='failed_not_scorable' and failed['source_and_timed_assets_SHA_verified']
    assert failed['no_full_process_time_or_PSNR_claim'] and failed['historical_log_compression_not_executed']
    guard=(pending/'wait_nccl_full209.py').read_text()
    assert guard.index('free>24*2**20')<guard.index('holds=[torch.empty')
    norm_root=root/'experiments/spatial-norm-221/world-model-r2'
    for name,digest in json.loads((norm_root/'source-hashes.json').read_text()).items():assert sha(norm_root/name)==digest,name
    assert sha(norm_root/'scripts/spatial_norm221_mode.py')==sha(root/'experiments/spatial-norm-221/cpu-r1/spatial_norm221_mode.py')
    norm_evidence=json.loads((root/'results/aggressive-20261028/spatial-norm-221/world-model-r2/summary.json').read_text())
    assert norm_evidence['clean'] and norm_evidence['passed'] and norm_evidence['ready_for_model']
    assert norm_evidence['baseline_source_count']==221 and not norm_evidence['full_V2_process_or_video_quality_qualification']
    for rank in norm_evidence['ranks']:
        assert len(rank['cases'])==6 and rank['RNG_unchanged'] and rank['baseline_restoration_bitwise']
        assert all(e['signedzero_bits'] for case in rank['cases'] for e in case['errors'])
        assert len(rank['pairs'])==6 and all(pair['saved_ms']>0 for pair in rank['pairs'])
        ctl=rank['attention_launches'];assert ctl['captured']==38 and ctl['no_new_collectives'] and ctl['no_temporal_GroupNorm_partition']
        assert ctl['adopted221']['temporal_block']['captured']==32
    for row in cases:
        records=re.findall(r'^PSNR case='+re.escape(row['case'])+r' psnr_db=([^ ]+) threshold_db=25 pass=(\w+)$',log,re.M)
        assert len(records)==1 and records[0][1]=='true'
    for prefix in ('RUN_SUMMARY ','RESULT_JSON=','ACCEPTED '):assert len(re.findall('^'+re.escape(prefix),log,re.M))==1
    assert not re.search(r'^FAILED:',log,re.M)
    from verify_all_feedback_evidence import verify as verify_all_feedback
    verify_all_feedback()
    from verify_fused_half_evidence import verify as verify_fused_half
    verify_fused_half()
    from verify_private222_evidence import verify as verify_private222
    verify_private222()
    from verify_nccl_tuning222_evidence import verify as verify_nccl_tuning
    verify_nccl_tuning()
    from verify_nonv2_ffn61_evidence import verify as verify_nonv2_ffn61
    verify_nonv2_ffn61()
    from verify_nonv2_temporal_block62_evidence import verify as verify_nonv2_temporal62
    verify_nonv2_temporal62()
    from verify_nonv2_all_blocks63_evidence import verify as verify_nonv2_all63
    verify_nonv2_all63()
    from verify_nonv2_other63_evidence import verify as verify_nonv2_other63
    verify_nonv2_other63()
    from verify_adopted222_profile import verify as verify_profile222
    verify_profile222()
    from verify_decision_temporal223_evidence import verify as verify_decision223
    verify_decision223()
    from verify_world_co223_evidence import verify as verify_world_co223
    verify_world_co223()
    from verify_nonv2_native_attention59_evidence import verify as verify_g1_attention
    verify_g1_attention()
    from verify_half_native_row_grid59_evidence import verify as verify_half_grid
    verify_half_grid()
    from verify_four_way226_evidence import verify as verify_four_way
    verify_four_way(root)
    from verify_fp16_joined_matched209 import verify as verify_matched_half
    verify_matched_half(root)
    from verify_crossstep211 import verify as verify_crossstep
    verify_crossstep(root)
    from verify_temporal_delta_gemm import verify as verify_delta
    verify_delta(root)
    from verify_g1_native_shared67 import verify as verify_g1_temporal
    verify_g1_temporal()
    from verify_g1_native_shared68 import verify as verify_g1_transport
    verify_g1_transport()
    from verify_nonv2_root64 import verify as verify_nonv2_transport
    verify_nonv2_transport()
    from verify_temporal_live_trials import verify as verify_temporal_live
    verify_temporal_live()
    from verify_nonv2_crossstep65 import verify as verify_temporal_dual
    verify_temporal_dual()
    from verify_nonv2_other_crossstep65 import verify as verify_other_temporal_dual
    verify_other_temporal_dual()
    from verify_temporal_prediction61 import verify as verify_temporal_prediction
    verify_temporal_prediction()
    from verify_nonv2_prediction65 import verify as verify_prediction_dual
    verify_prediction_dual()
    from verify_nonv2_other_prediction65 import verify as verify_prediction_other
    verify_prediction_other()
    from verify_nonv2_pencils_window65 import verify as verify_pencils_window
    verify_pencils_window()
    from verify_nonv2_late23_prediction65 import verify as verify_late23
    verify_late23()
    from verify_nonv2_dual_late23_prediction65 import verify as verify_dual_late23
    verify_dual_late23()
    from verify_nonv2_temporal_projection69 import verify as verify_temporal_projection
    verify_temporal_projection()
    from verify_v2_prediction211 import verify as verify_v2_prediction
    verify_v2_prediction()
    from verify_v2_late_prediction211 import verify as verify_v2_late
    verify_v2_late()
    verify_v2_late('both-late7-r4','public-late7-repro-cpu-r4',9,3)
    from verify_g1_prediction70 import verify as verify_g1_prediction
    verify_g1_prediction()
    from verify_g1_feature_trajectory71 import verify as verify_g1_trajectory
    verify_g1_trajectory()
    from verify_static_timestep65 import verify as verify_static_time
    verify_static_time()
    print('Verified: V2 CO225;65 two-refresh prediction Z1/dual/pencils;68 G1;all five dual GPU;fresh official103 log/JSON.')

if __name__=='__main__':main()
