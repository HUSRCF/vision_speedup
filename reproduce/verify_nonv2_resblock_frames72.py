"""Verify real model timing, frame ownership, source identity and full quality."""
import hashlib
import json
from pathlib import Path


def verify():
    repo=Path(__file__).resolve().parents[1]
    load=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    family=repo/'experiments/nonv2-resblock-frames72'
    evidence=repo/'results/aggressive-20261029/nonv2-resblock-frames72'
    base=repo/'experiments/nonv2-vae-all-display/unifolm'
    cpu=load(evidence/'public-repro-cpu-r1/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution']
    assert cpu['public_script_SHA256']==sha(repo/'reproduce/run_nonv2_resblock_frames72.py')
    prior=repo/'experiments/nonv2-all-blocks59/world-model-r1/scripts'
    for stage in ('decision','world'):
        frozen=family/(stage+'-model-r1')
        identity=load(frozen/'baseline-identity.json')
        assert identity['installed_baseline']=='qualified_original63' and identity['source_count']==59
        assert not identity['no_new_collective'] and identity['added_collectives']==22
        for name,h in dict(identity['manifest'],**identity['assets']).items():assert sha(base/name)==h
        for name,h in load(frozen/'source-hashes.json').items():assert sha(frozen/name)==h
        for name in ('nonv2_ffn59_mode.py','nonv2_temporal_block59_mode.py','nonv2_all_blocks59_mode.py'):
            assert sha(frozen/'scripts'/name)==sha(prior/name)
        r=load(evidence/(stage+'-model-r1/nccl-geglu-model-report.json'))
        assert r['passed'] and r['clean'] and r['ready_for_model'] and len(r['ranks'])==2
        for rank in r['ranks']:
            assert rank['baseline_is_original63_nonv2_FP16'] and rank['partitioned_master_tensors']==1004
            assert rank['RNG_unchanged'] and rank['baseline_restoration_bitwise'] and rank['zero_FFN_changes_baseline']
            assert len(rank['cases'])==len(rank['pairs'])==6
            assert all(e['gross_passed'] for c in rank['cases'] for e in c['errors'])
            assert all(p['saved_ms']>0 for p in rank['pairs'])
            ctl=rank['FFN_launches'];assert ctl['captured']==66
            assert ctl['temporal_block_captured']==34 and ctl['spatial_block_captured']==32
            res=ctl['resblock_frames'];assert res['modules']==res['captured']==22
            assert sum(x['local_send_bytes']for x in res['rows'])==120422400
            assert all(x['local_frames']==8 and x['temporal_full16']and x['added_collectives']==1 for x in res['rows'])
        for i in range(6):
            assert min(x['pairs'][i]['baseline_ms']for x in r['ranks'])>max(x['pairs'][i]['candidate_ms']for x in r['ranks'])
    frozen=family/'full66-r1';overlay=frozen/'production-overlay'
    parent=repo/'experiments/nonv2-late23-prediction65/full-r1'
    old=load(parent/'production-sources.json');manifest=load(frozen/'production-sources.json')
    runtime='src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    helper='src/unifolm_wma/utils/nonv2_resblock_frames72_mode.py'
    assert len(manifest)==66 and set(manifest)-set(old)=={helper}
    assert {n for n,h in old.items()if manifest[n]!=h}=={runtime}
    for name,h in manifest.items():
        p=overlay/name;assert sha(p if p.exists()else base/name)==h
    for name,h in load(frozen/'timing-asset-hashes.json').items():assert sha(overlay/name)==h
    for name,h in load(frozen/'guard-source-hashes.json').items():assert sha(frozen/name)==h
    packaged=(overlay/helper).read_text().replace('from .nonv2_all_blocks59_mode import','from nonv2_all_blocks59_mode import').replace('from .nonv2_ffn59_mode import','from nonv2_ffn59_mode import')
    assert packaged==(family/'decision-model-r1/scripts/nonv2_resblock_frames72_mode.py').read_text()
    entry='scripts/evaluation/world_model_interaction.py'
    assert (overlay/entry).read_bytes()==(parent/'production-overlay'/entry).read_bytes()
    cpu=load(evidence/'public-repro-cpu-r1/summary.json')
    assert len(cpu['trials'])==2 and all(x['all_four_CPU_checks_passed']for x in cpu['trials'])
    cpu=load(evidence/'full66-public-repro-cpu-r1/summary.json')
    assert cpu['passed']and cpu['no_GPU_execution']and cpu['source_count']==66
    assert cpu['exact_timed_manifest_and_assets']and cpu['frame_boundary_CPU_checked']
    assert cpu['lexical_cache_scope_passed']and cpu['predictor_affine3600_CPU_passed']
    assert cpu['public_script_SHA256']==sha(repo/'reproduce/run_nonv2_resblock66_full.py')
    run=evidence/'full66-r1';summary=load(run/'summary.json')
    assert summary['clean']and summary['quality_passed']and summary['workload_complete']and summary['rank_videos_exact']
    assert not summary['official_adopted'] and summary['process_wall_s']==104.738410
    assert summary['psnr_db']==26.21333418542557
    row=next(x for x in load(run/'result.json')['cases']if x['case']=='unitree_z1_dual_arm_cleanup_pencils')
    assert row['rounds']==8 and row['videos']==17 and row['psnr_passed']and row['profile_complete']and row['execution_ok']
    complete=load(run/'parallel/parallel-complete.json')
    assert complete['GPUs']==2 and complete['rank_videos_SHA256_equal']and len(complete['video_hashes'])==17
    assert complete['stage_calls']==dict(decision=400,world=400)
    for rank in (0,1):
        r=load(run/f'parallel/rank{rank}-controller.json')
        assert r['stage_captures']==dict(decision=36,world=36)and r['controller']['captured']==72
        assert {role:len(names)for role,names in r['resblock_graph_role_modules'].items()}==dict(decision_full=22,decision_reuse=1,world_full=22,world_reuse=1)
        assert r['controller']['resblock_frames']['captured']==46
        feature=r['feature_reuse'];assert feature['graph_count']==4 and feature['graph_prediction_selfcheck']['passed']
        assert feature['counts']==dict(decision_calls=400,decision_refresh=320,decision_reuse=80,world_calls=400,world_refresh=320,world_reuse=80)
        assert not feature['round_count_reduced']and not feature['sampler_step_count_reduced']
    repeats={}
    for label,is_candidate in (('candidate-repeat-r2',True),('baseline-repeat-r1',False)):
        identity=load(family/label/'source-identity.json')
        assert identity['manifest']==(manifest if is_candidate else old)
        expected_assets=load((frozen if is_candidate else parent)/'timing-asset-hashes.json')
        assert identity['asset_hashes']==expected_assets and identity['fresh_output_and_guard_state']
        run=evidence/label;summary=load(run/'summary.json');repeats[label]=summary
        assert summary['all_source_and_assets_SHA_unchanged']and summary['gpu_candidates']==['0','1']
        assert all(summary[k]for k in('clean','quality_passed','workload_complete','rank_videos_exact'))
        assert not summary['official_adopted']
        for name,h in identity['script_hashes'].items():assert sha(run/name)==h
        complete=load(run/'parallel/parallel-complete.json')
        expected=evidence/'full66-r1'if is_candidate else repo/'results/aggressive-20261029/nonv2-late23-prediction65/full-r1'
        assert complete['video_hashes']==load(expected/'parallel/parallel-complete.json')['video_hashes']
        assert complete['GPUs']==2 and complete['rank_videos_SHA256_equal']
        assert complete['stage_calls']==dict(decision=400,world=400)
        row=next(x for x in load(run/'result.json')['cases']if x['case']=='unitree_z1_dual_arm_cleanup_pencils')
        assert (row['rounds'],row['videos'])==(8,17)and row['psnr_passed']and row['execution_ok']
        assert row['process_wall_s']==summary['process_wall_s']and row['psnr_db']==summary['psnr_db']
    comparison=load(evidence/'complete-process-comparison.json')
    assert comparison['baseline_s']==repeats['baseline-repeat-r1']['process_wall_s']==94.368267
    assert comparison['candidate_s']==repeats['candidate-repeat-r2']['process_wall_s']==97.352066
    assert comparison['saved_s']==comparison['baseline_s']-comparison['candidate_s']<0
    adopted=load(repo/'experiments/nonv2-late23-prediction65/dual-accepted-outcome.json')
    for name,h in adopted['pair_hashes'].items():assert sha(repo/'results/history/official-before-resblock-tail66'/name)==h
    print('RESBLOCK72_VERIFIED original63_both_model_modes=true six_state_gross_screen=true full17_video_quality_passed=true matched_complete_regression2_984s=true formal_unchanged=true')


if __name__=='__main__':verify()
