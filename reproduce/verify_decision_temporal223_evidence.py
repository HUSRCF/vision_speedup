"""Check decision temporal routing qualification and complete trials when present."""
import hashlib,json
from pathlib import Path


def verify():
    root=Path(__file__).resolve().parents[1]
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    c=root/'experiments/decision-temporal-222';e=root/'results/aggressive-20261028/decision-temporal-222'
    original=load(root/'experiments/initial-boundary-221/production222/production-sources.json');assert len(original)==222
    model=c/'model-r1'
    for n,h in load(model/'source-hashes.json').items():assert sha(model/n)==h,n
    assert load(model/'baseline-production-identity.json')['manifest']==original
    report=load(e/'model-r1/summary.json');assert report['clean']and report['passed']and report['ready_for_model']and len(report['ranks'])==2
    for rank,row in enumerate(report['ranks']):
        assert row==load(e/f'model-r1/rank{rank}-report.json')
        assert len(row['cases'])==6 and all(x['signedzero_bits']for case in row['cases']for x in case['errors'])
        assert row['RNG_unchanged']and row['baseline_restoration_bitwise']and row['full_video_PSNR_unverified']and row['no_official_score']
        assert (row['Linear_master_tensors'],row['norm_master_tensors'],row['Conv2d_master_tensors'])==(570,782,192)
        assert len(row['pairs'])==6 and all(p['saved_ms']>0 for p in row['pairs'])
        ctl=row['attention_launches'];assert ctl['captured']==17 and ctl['additional_collectives_per_decision_forward']==17 and ctl['vendor_GEMM_descriptors_unchanged']and ctl['full5D_norm_and16_time_steps_preserved']
        assert len(ctl['rows'])==17 and all(x['collectives']==1 and x['complete16_time_steps']and x['full5D_GroupNorm_preserved']for x in ctl['rows'])
    mode='src/unifolm_wma/utils/decision_temporal223_mode.py';runtime='src/unifolm_wma/utils/nccl_parallel_runtime209.py'
    for variant,count in [('baseline',222),('candidate',223)]:
        folder=c/f'full-{variant}-r1';manifest=load(folder/'production-sources.json');assert len(manifest)==count
        if variant=='baseline':assert manifest==original
        else:
            assert set(manifest)-set(original)=={mode}and {n for n,h in original.items()if manifest[n]!=h}=={runtime}
            qualified=(model/'scripts/decision_temporal222_mode.py').read_text()
            production=(folder/'production-overlay'/mode).read_text()
            adapter='''
class ProductionController(Controller):
    @property
    def captures(self):return self.base.captures
    def report(self):
        report=super().report()
        return dict(report['adopted222'],decision_temporal={k:v for k,v in report.items()if k!='adopted222'})

'''
            expected=qualified.replace('\ndef install(model):\n','\n'+adapter+'\ndef install(model):\n').replace('active=Controller(model);return active','active=ProductionController(model);return active')
            assert production==expected
            proof=load(folder/'timing-proof.json');assert proof['router_model_source_sha256']==sha(model/'scripts/decision_temporal222_mode.py')and proof['original209_math_sources_unchanged']and proof['production_adapter_only_changes_report_and_capture_accounting']
            for n,h in manifest.items():
                p=folder/'production-overlay'/n
                if p.exists():assert sha(p)==h,n
        for n,h in load(folder/'guard-source-hashes.json').items():assert sha(folder/n)==h,n
        cpu=(folder/'cpu-preflight.log').read_text();assert cpu.startswith('NCCL_PARALLEL_CPU_CHECK ')
        cpu=json.loads(cpu.split(' ',1)[1]);assert cpu['passed']and cpu['no_GPU_initialized']and cpu['no_pre_timer_torch_import']
        out=e/f'full-{variant}-r1'
        if not(out/'summary.json').exists():continue
        summary=load(out/'summary.json');assert summary['clean']and summary['workload_complete']and summary['rank_videos_exact']and summary['verified_source_count']==count
        case='unitree_z1_dual_arm_stackbox_v2';row=next(x for x in load(out/'result.json')['cases']if x['case']==case)
        assert row['rounds']==11 and row['videos']==23 and row['execution_ok']and row['profile_complete']and row['psnr_passed']==summary['quality_passed']
        assert row['process_wall_s']==summary['process_wall_s']and row['psnr_db']==summary['psnr_db']
        assert load(out/'profiles/execution.json')[case]['exit_code']==0
        parallel=load(out/'parallel/parallel-complete.json');assert parallel['stage_calls']==dict(decision=550,world=550)and parallel['rank_videos_SHA256_equal']and parallel['video_count']==23
        for rank in(0,1):
            ctl=load(out/f'parallel/rank{rank}-controller.json');assert ctl['stage_captures']==dict(decision=16,world=32)
            if variant=='candidate':assert ctl['controller']['decision_temporal']['captured']==17
        comparison=load(out/'baseline-video-comparison.json');assert len(comparison['current_hashes'])==len(comparison['baseline_hashes'])==23
        assert comparison['all23_SHA256_equal']==summary['baseline_videos_exact']
    verify_adoption(root)
    print('Adopted223:original222 baseline,vendor descriptors,six-state bitwise pairs,matched full AB and fresh official103 verified.')


def verify_adoption(root):
    import ast
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    code=root/'experiments/decision-temporal-222';ev=root/'results/aggressive-20261028/decision-temporal-222'
    outcome=load(code/'accepted-outcome.json');assert outcome['official_adopted']and outcome['all103_officially_validated']and outcome['source_count']==223 and outcome['GPUs']==2
    assert outcome['V2_s']==265.216325 and outcome['PSNR_dB']==29.675324786148643 and outcome['five_case_sum_s']==778.585395
    assert outcome['one_complete_process_AB_not_six']and outcome['six_model_pairs']and outcome['all23_videos_SHA_equal']and outcome['original209_math_unchanged']and outcome['vendor_GEMM_descriptors_unchanged']and outcome['full5D_GroupNorm_and16T_preserved']and outcome['original_process_timer_unchanged']and outcome['additional_collectives_per_decision_forward']==17 and not outcome['website_uploaded']
    frozen=code/'full-candidate-r1';erratum=load(frozen/'timing-proof-erratum.json')
    assert erratum['raw_timed_proof_SHA256']==sha(frozen/'timing-proof.json')and erratum['field']=='no_new_collectives'and erratum['raw_value']is True and erratum['correct_candidate_value']is False and erratum['additional_collectives_per_decision_forward']==17 and erratum['world_additional_collectives']==0 and erratum['original_timed_proof_not_rewritten']
    pair=load(ev/'complete-process-AB.json');assert pair['same_GPU_pair']and pair['physical_GPUs']==['0','1']and pair['same_guard_scripts']and pair['same_preload_binary']and pair['same_private_tmpfs_storage']and pair['original_evaluator_timer_preserved']and pair['one_full_AB_not_six_full_process_pairs']
    complete=[]
    for variant in('baseline','candidate'):
        folder=code/f'full-{variant}-r1';out=ev/f'full-{variant}-r1';summary=load(out/'summary.json');assert summary['quality_passed']and summary['baseline_videos_exact']
        raw=load(out/'result.json');row=next(x for x in raw['cases']if x['case']=='unitree_z1_dual_arm_stackbox_v2');complete.append(row)
        assert row['process_wall_s']==pair['complete'][variant]['process_wall_s']and row['psnr_db']==pair['complete'][variant]['psnr_db']
        assert load(out/'nccl-full-report.json')['physical_GPUs']==pair['physical_GPUs']
        profile=load(out/'profiles/unitree_z1_dual_arm_stackbox_v2.json')
        for phase in('decision_generation','simulation_generation'):assert profile['stages'][phase]==pair['complete'][variant][phase]
        original=load(root/'unifolm/world-head-filter-cache-production-sources.json')
        manifest=load(folder/'production-sources.json');assert all(manifest[n]==h for n,h in original.items())
    assert abs(complete[0]['process_wall_s']-complete[1]['process_wall_s']-pair['same_GPU_full_AB_saved_s'])<1e-6 and pair['same_GPU_full_AB_saved_s']==outcome['same_GPU_full_AB_saved_s']
    for n in('wait_nccl_full209.py','retry_nccl_full221.py','check_dual_gpu_graph.py'):assert sha(code/'full-baseline-r1'/n)==sha(code/'full-candidate-r1'/n)
    assert load(code/'full-baseline-r1/flags.json')==load(code/'full-candidate-r1/flags.json')
    historical=root/'results/history/official-other63-before-decision223';official=ev/'official103'
    for n,h in outcome['previous_pair_sha256'].items():assert sha(historical/n)==h
    for n,h in outcome['official_pair_sha256'].items():assert sha(official/n)==sha(root/'results'/n)==h
    for n,h in load(official/'script-hashes.json').items():assert sha(official/n)==h
    result=load(official/'result.json');assert result==load(root/'results/result.json')and result['status']=='accepted'and result['psnr_validation']['video_count']==103
    previous={r['case']:r for r in load(historical/'result.json')['cases']}
    for row in result['cases']:
        old=previous[row['case']]
        if row['case']=='unitree_z1_dual_arm_stackbox_v2':assert row['process_wall_s']==outcome['V2_s']and row['psnr_db']==outcome['PSNR_dB']and old['process_wall_s']==outcome['previous_V2_s']
        else:assert row['process_wall_s']==old['process_wall_s']and row['psnr_db']==old['psnr_db']
    assert abs(sum(r['process_wall_s']for r in result['cases'])-outcome['five_case_sum_s'])<1e-6
    links=load(official/'immutable-video-copies.json');assert links['count']==103 and links['ordinary_files']and links['original_outputs_unchanged']and links['SHA_verified_same_FS_links_or_cross_FS_copies']
    public=load(ev/'public-repro-cpu-r1/summary.json');assert public['passed']and public['no_GPU_execution']and public['source_count']==223 and public['exact_timed_manifest_and_assets']and public['public_script_SHA256']==sha(root/'reproduce/run_v2_decision223.py')
    gateway=load(ev/'public-gateway-cpu-r1/summary.json');assert gateway['passed']and gateway['no_GPU_execution']and gateway['source_count']==223 and gateway['exact_timed_manifest_and_assets']and gateway['default_entry_rebuilds_adopted223']
    assert gateway['public_gateway_SHA256']==sha(root/'reproduce/run_v2_parallel.py')and gateway['public_runner_SHA256']==public['public_script_SHA256']and gateway['home_free_before']==gateway['home_free_after']
    parser=ast.parse((root/'reproduce/run_v2_parallel.py').read_text());assert any(isinstance(n,ast.keyword)and n.arg=='default'and isinstance(n.value,ast.Constant)and n.value.value=='decision-temporal223'for n in ast.walk(parser))

if __name__=='__main__':verify()
