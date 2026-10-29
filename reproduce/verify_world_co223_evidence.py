"""Verify CO-sharded Winograd qualification without claiming a formal score."""
import hashlib,json,statistics
from pathlib import Path


def verify():
    root=Path(__file__).resolve().parents[1]
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    code=root/'experiments/world-co-split223'
    evidence=root/'results/aggressive-20261028/world-co-split223'
    baseline=load(root/'experiments/decision-temporal-222/full-candidate-r1/production-sources.json')
    assert len(baseline)==223
    for label,count in [('conv-r1',6),('conv-r2',28),('model-r1',None)]:
        folder=code/label
        for n,h in load(folder/'source-hashes.json').items():assert sha(folder/n)==h,n
        assert load(folder/'baseline-production-identity.json')['manifest']==baseline
        report=load(evidence/label/'nccl-geglu-model-report.json')
        assert report['passed']and report['clean']and report['no_official_score']
        assert len(report['ranks'])==2
        for rank,row in enumerate(report['ranks']):
            assert row==load(evidence/label/f'rank{rank}-report.json')
            assert row['no_official_score']
            if count is not None:
                assert row['synthetic_only']and row['all_full_output_dynamic_cases_bitwise']
                assert len(row['rows'])==count
                for item in row['rows']:
                    assert len(item['cases'])==6 and all(x['all16_full_output_signedzero_bits']and x['finite']for x in item['cases'])
                    for flag in ['all6_dynamic_full_outputs_signedzero_bits','original_native_and_numerical_restorations_bitwise','RNG_unchanged','original_CUBIN_constexpr_and_K_order','original_native_transforms_bias_and_version_refresh','pack_one_allgather_and_layout_restore_in_timing','output_workspace_poisoned_each_case','no_workspace_tensor_views_or_operand_exports']:assert item[flag],flag
                    assert len(item['pairs'])==6
                    assert item['median_saved_ms']==statistics.median(p['saved_ms']for p in item['pairs'])
            else:
                assert row['stage']=='world'and row['baseline_is_adopted_parallel_routing']
                assert row['RNG_unchanged']and row['baseline_restoration_bitwise']and row['full_video_PSNR_unverified']
                assert (row['Linear_master_tensors'],row['norm_master_tensors'],row['Conv2d_master_tensors'])==(570,782,192)
                assert len(row['cases'])==6 and all(x['signedzero_bits']for case in row['cases']for x in case['errors'])
                assert len(row['pairs'])==6 and all(p['saved_ms']>0 for p in row['pairs'])
                assert row['median_saved_ms']==statistics.median(p['saved_ms']for p in row['pairs'])
                ctl=row['attention_launches'];assert ctl['captured']==9 and ctl['extra_world_collectives_per_forward']==9 and ctl['new_kernels']==0
                assert len(ctl['rows'])==9 and all(x['collectives']==1 for x in ctl['rows'])
                assert sum(x['local_MiB']for x in ctl['rows'])==187.5
                coverage=ctl['native_cache_coverage'];assert len(coverage)==1
                assert coverage[0]['selected']==9 and coverage[0]['original_owners']==23 and coverage[0]['original_head_caches']==45 and coverage[0]['original_strict_CUBIN_and_cache_checks_completed_first']
    choice=load(code/'model-r1/selection.json');assert choice['selected_owner_count']==9
    assert choice['minimum_net_gain_threshold_ms']==0.05
    measured={}
    for rank in(0,1):
        for row in load(evidence/f'conv-r2/rank{rank}-report.json')['rows']:measured.setdefault(tuple(row['geometry']),[]).append(row)
    assert len(measured)==14 and all(len(v)==4 for v in measured.values())
    for item in choice['selected']:
        rows=measured[tuple(item['geometry'])]
        minimum=min(r['median_saved_ms']for r in rows)
        assert minimum==item['minimum_geometry_net_saved_ms']and minimum>0.05
        assert all(r['all6_pairs_faster']for r in rows)
    assert abs(sum(x['minimum_geometry_net_saved_ms']for x in choice['selected'])-choice['sum_minimum_synthetic_net_saved_ms'])<1e-9
    assert len(choice['selected'])+len(choice['excluded'])==23
    public=load(evidence/'public-repro-cpu-r1/summary.json')
    assert public['passed']and public['no_GPU_execution']and public['source_count']==225 and public['exact_timed_manifest_and_assets']
    assert public['public_script_SHA256']==sha(root/'reproduce/run_v2_world_co225.py')and public['home_free_before']==public['home_free_after']
    runtime='src/unifolm_wma/utils/nccl_parallel_runtime209.py'
    core='src/unifolm_wma/utils/world_co_split225.py'
    mode='src/unifolm_wma/utils/world_co_split225_mode.py'
    for variant,count in [('baseline',223),('candidate',225)]:
        folder=code/f'full-{variant}-r1';manifest=load(folder/'production-sources.json')
        assert len(manifest)==count
        if variant=='baseline':assert manifest==baseline
        else:
            assert set(manifest)-set(baseline)=={core,mode}
            assert {n for n,h in baseline.items()if manifest[n]!=h}=={runtime}
            for n,h in manifest.items():
                p=folder/'production-overlay'/n
                if p.exists():assert sha(p)==h,n
            assert sha(folder/'production-overlay'/core)==sha(code/'model-r1/scripts/world_co_split223.py')
            qualified=(code/'model-r1/scripts/world_co_split223_mode.py').read_text()
            expected=qualified.replace('from world_co_split223 import','from .world_co_split225 import')
            expected=expected.replace("scope=json.loads((Path(__file__).resolve().parent.parent/'selection.json').read_text())",'scope='+repr(choice))
            adapter='''
class ProductionController(Controller):
    @property
    def captures(self):return self.base.captures
    def report(self):
        report=super().report()
        return dict(report['adopted223'],world_CO={k:v for k,v in report.items()if k!='adopted223'})

'''
            expected=expected.replace('\ndef install(model):\n','\n'+adapter+'\ndef install(model):\n').replace('active=Controller(model);return active','active=ProductionController(model);return active')
            assert (folder/'production-overlay'/mode).read_text()==expected
            proof=load(folder/'timing-proof.json')
            assert proof['no_new_collectives']is False and proof['additional_world_collectives_per_forward']==9 and proof['source_count']==225 and proof['new_JIT_kernels']==0
        for n,h in load(folder/'guard-source-hashes.json').items():assert sha(folder/n)==h,n
        cpu=json.loads((folder/'cpu-preflight.log').read_text().split(' ',1)[1])
        assert cpu['passed']and cpu['no_GPU_initialized']and cpu['no_pre_timer_torch_import']
        out=evidence/f'full-{variant}-r1'
        if not(out/'summary.json').exists():continue
        summary=load(out/'summary.json');assert summary['clean']and summary['workload_complete']and summary['verified_source_count']==count
        case='unitree_z1_dual_arm_stackbox_v2'
        row=next(x for x in load(out/'result.json')['cases']if x['case']==case)
        assert row['rounds']==11 and row['videos']==23 and row['execution_ok']and row['profile_complete']
        assert row['process_wall_s']==summary['process_wall_s']and row['psnr_db']==summary['psnr_db']and row['psnr_passed']==summary['quality_passed']
        parallel=load(out/'parallel/parallel-complete.json')
        assert parallel['stage_calls']==dict(decision=550,world=550)and parallel['rank_videos_SHA256_equal']and parallel['video_count']==23
        compare=load(out/'baseline-video-comparison.json');assert len(compare['current_hashes'])==len(compare['baseline_hashes'])==23
        assert compare['all23_SHA256_equal']==summary['baseline_videos_exact']
        for rank in(0,1):
            ctl=load(out/f'parallel/rank{rank}-controller.json');assert ctl['stage_captures']==dict(decision=16,world=32)
            assert ctl['controller']['decision_temporal']['captured']==17
            if variant=='candidate':assert ctl['controller']['world_CO']['captured']==9
    outcome=code/'accepted-outcome.json'
    if outcome.exists():verify_adoption(root,load(outcome))
    print('CO223 qualification:14 geometries/two layouts/two ranks/six states,9 selected owners and six positive full-model pairs verified; no formal score inferred.')


def verify_adoption(root,outcome):
    import ast
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    evidence=root/'results/aggressive-20261028/world-co-split223';code=root/'experiments/world-co-split223'
    assert outcome['official_adopted']and outcome['all103_officially_validated']and outcome['source_count']==225 and outcome['GPUs']==2 and not outcome['website_uploaded']
    assert outcome['additional_world_collectives_per_forward']==9 and outcome['existing_decision_collectives_preserved']==17 and outcome['original209_math_unchanged']and outcome['original_CUBIN_constexpr_batch_strides_K_order_preserved']
    pair=load(evidence/'complete-process-AB.json');assert pair['same_GPU_pair']and pair['physical_GPUs']==['0','1']and pair['same_guard_scripts']and pair['same_preload_binary']and pair['same_private_tmpfs_storage']and pair['original_evaluator_timer_preserved']and pair['one_full_AB_not_six_full_process_pairs']
    assert pair['official_adopted']and pair['baseline_source_count']==223 and pair['candidate_source_count']==225
    baseline=pair['complete']['baseline'];candidate=pair['complete']['candidate']
    assert candidate['process_wall_s']==outcome['V2_s']and baseline['process_wall_s']==outcome['same_GPU_full_AB_baseline_s']
    assert abs(baseline['process_wall_s']-candidate['process_wall_s']-outcome['same_GPU_full_AB_saved_s'])<1e-6
    assert pair['same_GPU_full_AB_saved_s']==outcome['same_GPU_full_AB_saved_s']and pair['same_GPU_full_AB_saved_s']>0
    for variant in('baseline','candidate'):
        report=load(evidence/f'full-{variant}-r1/nccl-full-report.json');assert report['physical_GPUs']==['0','1']and report['clean']
        profile=load(evidence/f'full-{variant}-r1/profiles/unitree_z1_dual_arm_stackbox_v2.json')
        for phase in('decision_generation','simulation_generation'):assert profile['stages'][phase]==pair['complete'][variant][phase]
    official=evidence/'official103';history=root/'results/history/official223-before-world-co225'
    for n,h in outcome['previous_pair_sha256'].items():assert sha(history/n)==h
    current_archive=root/'results/history/official-before-g1-native68'
    for n,h in outcome['official_pair_sha256'].items():assert sha(official/n)==sha(current_archive/n)==h
    for n,h in load(official/'script-hashes.json').items():assert sha(official/n)==h
    result=load(official/'result.json');assert result==load(current_archive/'result.json')and result['status']=='accepted'and result['psnr_validation']['video_count']==103
    old={r['case']:r for r in load(history/'result.json')['cases']}
    for row in result['cases']:
        previous=old[row['case']]
        if row['case']=='unitree_z1_dual_arm_stackbox_v2':assert row['process_wall_s']==outcome['V2_s']and row['psnr_db']==outcome['PSNR_dB']==previous['psnr_db']and previous['process_wall_s']==outcome['previous_V2_s']
        else:assert row['process_wall_s']==previous['process_wall_s']and row['psnr_db']==previous['psnr_db']
    assert abs(sum(r['process_wall_s']for r in result['cases'])-outcome['five_case_sum_s'])<1e-6
    copies=load(official/'immutable-video-copies.json');assert copies['count']==103 and copies['ordinary_files']and copies['original_outputs_unchanged']and copies['SHA_verified_same_FS_links_or_cross_FS_copies']
    gateway=load(evidence/'public-gateway-cpu-r1/summary.json');assert gateway['passed']and gateway['no_GPU_execution']and gateway['source_count']==225 and gateway['exact_timed_manifest_and_assets']
    assert gateway['public_script_SHA256']==sha(root/'reproduce/run_v2_parallel.py')and gateway['public_runner_SHA256']==sha(root/'reproduce/run_v2_world_co225.py')and gateway['home_free_before']==gateway['home_free_after']
    parser=ast.parse((root/'reproduce/run_v2_parallel.py').read_text());assert any(isinstance(n,ast.keyword)and n.arg=='default'and isinstance(n.value,ast.Constant)and n.value.value=='world-co225'for n in ast.walk(parser))
    print('Adopted225:matched complete process AB,unchanged23 videos,fresh official103 and public default CPU preparation verified.')


if __name__=='__main__':verify()
