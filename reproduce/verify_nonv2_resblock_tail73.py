"""Verify real model timing, frame ownership, source identity and full quality."""
import hashlib
import json
from pathlib import Path


def verify():
    repo=Path(__file__).resolve().parents[1]
    load=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    family=repo/'experiments/nonv2-resblock-tail73'
    evidence=repo/'results/aggressive-20261029/nonv2-resblock-tail73'
    base=repo/'experiments/nonv2-vae-all-display/unifolm'
    cpu=load(evidence/'public-repro-cpu-r1/summary.json')
    assert cpu['passed'] and cpu['no_GPU_execution']
    assert cpu['public_script_SHA256']==sha(repo/'reproduce/run_nonv2_resblock_tail73.py')
    prior=repo/'experiments/nonv2-all-blocks59/world-model-r1/scripts'
    for stage in ('decision','world'):
        frozen=family/(stage+'-model-r1')
        identity=load(frozen/'baseline-identity.json')
        assert identity['installed_baseline']=='qualified_original63' and identity['source_count']==59
        assert not identity['no_new_collective'] and identity['added_collectives']==5
        for name,h in dict(identity['manifest'],**identity['assets']).items():assert sha(base/name)==h
        for name,h in load(frozen/'source-hashes.json').items():assert sha(frozen/name)==h
        for name in ('nonv2_ffn59_mode.py','nonv2_temporal_block59_mode.py','nonv2_all_blocks59_mode.py'):
            assert sha(frozen/'scripts'/name)==sha(prior/name)
        r=load(evidence/(stage+'-model-r1/nccl-geglu-model-report.json'))
        assert r['passed'] and r['clean'] and r['ready_for_model'] and len(r['ranks'])==2
        for rank in r['ranks']:
            assert rank['baseline_is_original63_nonv2_FP16'] and rank['partitioned_master_tensors']==816
            assert rank['RNG_unchanged'] and rank['baseline_restoration_bitwise'] and rank['zero_FFN_changes_baseline']
            assert len(rank['cases'])==len(rank['pairs'])==6
            assert all(e['signedzero_bits'] for c in rank['cases'] for e in c['errors'])
            assert all(p['saved_ms']>0 for p in rank['pairs'])
            ctl=rank['FFN_launches'];assert ctl['captured']==66
            assert ctl['temporal_block_captured']==34 and ctl['spatial_block_captured']==32
            res=ctl['resblock_frames'];assert res['modules']==res['captured']==5
            assert sum(x['local_send_bytes']for x in res['rows'])==52428800
            assert all(x['local_frames']==8 and x['temporal_full16']and x['added_collectives']==1 for x in res['rows'])
        for i in range(6):
            assert min(x['pairs'][i]['baseline_ms']for x in r['ranks'])>max(x['pairs'][i]['candidate_ms']for x in r['ranks'])
    frozen=family/'full66-r1';overlay=frozen/'production-overlay'
    parent=repo/'experiments/nonv2-late23-prediction65/full-r1'
    old=load(parent/'production-sources.json');manifest=load(frozen/'production-sources.json')
    runtime='src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    helper='src/unifolm_wma/utils/nonv2_resblock_tail73_mode.py'
    assert len(manifest)==66 and set(manifest)-set(old)=={helper}
    assert {n for n,h in old.items()if manifest[n]!=h}=={runtime}
    assert load(frozen/'trial.json')['per_role_expected_gathers']==dict(decision_full=38,decision_reuse=4,world_full=38,world_reuse=4)
    for name,h in manifest.items():
        p=overlay/name;assert sha(p if p.exists()else base/name)==h
    for name,h in load(frozen/'timing-asset-hashes.json').items():assert sha(overlay/name)==h
    for name,h in load(frozen/'guard-source-hashes.json').items():assert sha(frozen/name)==h
    packaged=(overlay/helper).read_text().replace('from .nonv2_all_blocks59_mode import','from nonv2_all_blocks59_mode import').replace('from .nonv2_ffn59_mode import','from nonv2_ffn59_mode import')
    assert packaged==(family/'decision-model-r1/scripts/nonv2_resblock_tail73_mode.py').read_text()
    all22=(repo/'experiments/nonv2-resblock-frames72/decision-model-r1/scripts/nonv2_resblock_frames72_mode.py').read_text()
    expected=all22.replace('            if type(module) is not ResBlock:', "            if type(module) is not ResBlock or name not in {'output_blocks.7.0','output_blocks.8.0','output_blocks.9.0','output_blocks.10.0','output_blocks.11.0'}:").replace('assert len(self.modules) == 22','assert len(self.modules) == 5')
    assert packaged==expected
    entry='scripts/evaluation/world_model_interaction.py'
    assert (overlay/entry).read_bytes()==(parent/'production-overlay'/entry).read_bytes()
    cpu=load(evidence/'public-repro-cpu-r1/summary.json')
    assert len(cpu['trials'])==2 and all(x['all_four_CPU_checks_passed']for x in cpu['trials'])
    cpu=load(evidence/'full66-public-repro-cpu-r1/summary.json')
    assert cpu['passed']and cpu['no_GPU_execution']and cpu['source_count']==66
    assert cpu['exact_timed_manifest_and_assets']and cpu['frame_boundary_CPU_checked']
    assert cpu['lexical_cache_scope_passed']and cpu['predictor_affine3600_CPU_passed']
    assert cpu['public_script_SHA256']==sha(repo/'reproduce/run_nonv2_resblock_tail66_full.py')
    run=evidence/'full66-r1';summary=load(run/'summary.json')
    assert summary['clean']and summary['quality_passed']and summary['workload_complete']and summary['rank_videos_exact']
    assert not summary['official_adopted'] and summary['process_wall_s']==92.232482
    assert summary['psnr_db']==26.79860623954979
    row=next(x for x in load(run/'result.json')['cases']if x['case']=='unitree_z1_dual_arm_cleanup_pencils')
    assert row['rounds']==8 and row['videos']==17 and row['psnr_passed']and row['profile_complete']and row['execution_ok']
    complete=load(run/'parallel/parallel-complete.json')
    assert complete['GPUs']==2 and complete['rank_videos_SHA256_equal']and len(complete['video_hashes'])==17
    assert complete['stage_calls']==dict(decision=400,world=400)
    assert complete['video_hashes']==load(repo/'results/aggressive-20261029/nonv2-late23-prediction65/full-r1/parallel/parallel-complete.json')['video_hashes']
    for rank in (0,1):
        r=load(run/f'parallel/rank{rank}-controller.json')
        assert r['stage_captures']==dict(decision=36,world=36)and r['controller']['captured']==72
        assert {role:len(names)for role,names in r['resblock_graph_role_modules'].items()}==dict(decision_full=5,decision_reuse=1,world_full=5,world_reuse=1)
        assert {role:len(names)+len(r['resblock_graph_role_modules'][role])for role,names in r['graph_role_modules'].items()}==dict(decision_full=38,decision_reuse=4,world_full=38,world_reuse=4)
        assert r['controller']['resblock_frames']['captured']==12
        feature=r['feature_reuse'];assert feature['graph_count']==4 and feature['graph_prediction_selfcheck']['passed']
        assert feature['counts']==dict(decision_calls=400,decision_refresh=320,decision_reuse=80,world_calls=400,world_refresh=320,world_reuse=80)
        assert not feature['round_count_reduced']and not feature['sampler_step_count_reduced']
    comparison=load(evidence/'complete-process-comparison.json')
    assert comparison['GPUs']==['0','1'] and comparison['one_candidate_two_bracketing_controls']
    assert comparison['candidate_s']==92.232482
    assert comparison['baseline_before_s']==94.368267 and comparison['baseline_after_s']==93.983744
    assert comparison['saved_vs_before_s']==comparison['baseline_before_s']-comparison['candidate_s']>0
    assert comparison['saved_vs_after_s']==comparison['baseline_after_s']-comparison['candidate_s']>0
    after=evidence/'baseline-after-full-r2';identity=load(family/'baseline-after-full-r2/source-identity.json')
    assert identity['manifest']==old and identity['asset_hashes']==load(parent/'timing-asset-hashes.json')
    for name,h in identity['script_hashes'].items():assert sha(after/name)==h
    assert load(after/'parallel/parallel-complete.json')['video_hashes']==complete['video_hashes']
    assert load(after/'summary.json')['process_wall_s']==comparison['baseline_after_s']
    adopted=load(family/'accepted-outcome.json');official=evidence/'official103'
    assert adopted['official_adopted']and adopted['official103_accepted']and adopted['local_GPU_pair_updated']
    assert adopted['source_count']==66 and adopted['GPUs']==2 and not adopted['website_uploaded']
    assert adopted['case_sum_s']==735.397241 and adopted['previous_case_sum_s']==738.805454
    assert adopted['formal_saved_s']==3.408213 and adopted['all_rounds_and50_steps_preserved']
    assert adopted['all103_video_SHA_unchanged']and adopted['selected17_formal_SHA_exact']
    assert adopted['selected_video_count']==17 and adopted['unchanged_video_count']==86
    for name,h in adopted['assembly_script_hashes'].items():assert sha(official/name)==h
    fresh=load(official/'validated/result.json')
    assert fresh==load(repo/'results/result.json') and fresh['status']=='accepted'
    assert fresh['psnr_validation']['video_count']==103
    history=repo/'results/history/official-before-resblock-tail66'
    for name,h in adopted['pair_hashes'].items():assert sha(repo/'results'/name)==sha(official/'validated'/name)==h
    for name,h in adopted['previous_pair_hashes'].items():assert sha(history/name)==h
    previous=load(history/'result.json')
    for row in fresh['cases']:
        before=next(x for x in previous['cases']if x['case']==row['case'])
        if row['case']=='unitree_z1_dual_arm_cleanup_pencils':
            assert row['process_wall_s']==92.232482 and row['psnr_db']==before['psnr_db']==26.79860623954979
        else:assert row==before
    links=load(official/'validated/immutable-video-links.json')
    old_links=load(repo/'results/aggressive-20261029/nonv2-late23-prediction65/dual-official103/validated/immutable-video-links.json')
    normalize=lambda x:{p.rsplit('/outputs/',1)[1]:h for p,h in x['hashes'].items()}
    assert len(normalize(links))==103 and normalize(links)==normalize(old_links)
    assert links['ordinary_files']and links['original_outputs_unchanged']and links['no_GPU_rerun_or_timer_rewrite']
    print("TAIL73_VERIFIED both_stage_six_state_bitwise=true full17_formal_SHA_exact=true source66=true bracketed_complete_faster=true fresh103_accepted=true all103_SHA_unchanged=true")


if __name__=='__main__':verify()
