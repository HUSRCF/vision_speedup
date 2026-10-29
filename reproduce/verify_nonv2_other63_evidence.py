"""Check per-case math identity, full workloads and the adopted103 bundle."""
import hashlib,json
from pathlib import Path

CASES={'dual':('unitree_z1_dual_arm_stackbox',7),'pencils':('unitree_z1_dual_arm_cleanup_pencils',8),'g1':('unitree_g1_pack_camera',11)}


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify():
    root=Path(__file__).resolve().parents[1];code=root/'experiments/nonv2-other-blocks63';ev=root/'results/aggressive-20261028/nonv2-other-blocks63'
    base=root/'experiments/nonv2-vae-all-display/unifolm';original=load(base/'nonv2-aggressive-production-sources.json');assert len(original)==59
    z1=root/'experiments/nonv2-all-blocks59/full63-r1';reference=load(z1/'production-sources.json')
    runtime='src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    records={}
    for alias,(case,rounds)in CASES.items():
        folder=code/(alias+'-full-r1');out=ev/(alias+'-full-r1');manifest=load(folder/'production-sources.json')
        assert len(manifest)==63 and all(manifest[n]==h for n,h in original.items())
        assert {n for n in reference if manifest[n]!=reference[n]}=={runtime}
        for n,h in manifest.items():
            p=folder/'production-overlay'/n;assert sha(p if p.exists()else base/n)==h,n
        assert(folder/'production-overlay'/runtime).read_text()==(z1/'production-overlay'/runtime).read_text().replace("'unitree_z1_stackbox'",repr(case))
        for n,h in load(folder/'guard-source-hashes.json').items():assert sha(folder/n)==h
        for n,h in load(folder/'timing-asset-hashes.json').items():
            if n=='nccl-entry-timing-proof.json':p=folder/'timing-proof.json'
            else:
                p=folder/'production-overlay'/n
                if not p.exists():p=base/n
            assert sha(p)==h,n
        flags=load(folder/'flags.json');assert flags==dict(load(base.parent/(case+'-flags.json')),UNIFOLM_NCCL_PARALLEL='nonv2-all-blocks')
        if alias=='g1':assert flags['UNIFOLM_FP32_FEEDBACK_HEADS']=='1'and flags['UNIFOLM_SIM_HEAD_TF32']=='0'
        proof=load(folder/'timing-proof.json');assert proof['dataset_case']==case and proof['official_rounds']==rounds and proof['Z1_PSNR_not_inherited']and proof['same_three_math_helpers_as_adopted_Z1']
        summary=load(out/'summary.json');assert summary['clean']and summary['workload_complete']and summary['rank_videos_exact']and summary['case']==case
        row=next(r for r in load(out/'result.json')['cases']if r['case']==case)
        assert row['rounds']==rounds and row['videos']==2*rounds+1 and row['execution_ok']and row['profile_complete']
        assert row['psnr_passed']==summary['quality_passed']and row['psnr_db']==summary['psnr_db']and row['process_wall_s']==summary['process_wall_s']
        assert load(out/'profiles/execution.json')[case]['exit_code']==0
        profile=load(out/'profiles'/f'{case}.json');assert profile['total_process_wall_s']==row['process_wall_s']
        assert all(profile['stages'][stage]['count']==rounds for stage in('decision_generation','simulation_generation'))
        parallel=load(out/'parallel/parallel-complete.json');assert parallel['stage_calls']==dict(decision=50*rounds,world=50*rounds)and parallel['rank_videos_SHA256_equal']and len(parallel['video_hashes'])==2*rounds+1
        for rank in(0,1):
            ctl=load(out/f'parallel/rank{rank}-controller.json');assert ctl['stage_captures']==dict(decision=33,world=33)
            report=ctl['controller'];assert report['captured']==66 and report['spatial_block_captured']==32 and report['temporal_block_captured']==34 and all(r['collectives']==1 for r in report['rows'])
        records[case]=row
    outcome=load(code/'accepted-outcome.json');assert outcome['official_adopted']and outcome['all103_officially_validated']and not outcome['website_uploaded']
    before=root/'results/history/official-z1-blocks63-before-other63'
    for n,h in outcome['previous_pair_sha256'].items():assert sha(before/n)==h
    official=ev/'official103'
    for n,h in load(official/'script-hashes.json').items():assert sha(official/n)==h
    historical=root/'results/history/official-other63-before-decision223'
    for n,h in outcome['official_pair_sha256'].items():assert sha(official/n)==sha(historical/n)==h
    result=load(official/'result.json');assert result['status']=='accepted'and result['psnr_validation']['video_count']==103 and result==load(historical/'result.json')
    current={r['case']:r for r in load(root/'results/history/official-before-g1-native68/result.json')['cases']}
    for row in result['cases']:
        if row['case']!='unitree_z1_dual_arm_stackbox_v2':
            assert current[row['case']]['process_wall_s']==row['process_wall_s']and current[row['case']]['psnr_db']==row['psnr_db']
    previous={r['case']:r for r in load(before/'result.json')['cases']}
    for row in result['cases']:
        case=row['case']
        if case in outcome['cases']:
            expected=records[case];assert expected['psnr_passed']and expected['process_wall_s']<previous[case]['process_wall_s']
            assert row['process_wall_s']==expected['process_wall_s']and row['psnr_db']==expected['psnr_db']
        else:assert row['process_wall_s']==previous[case]['process_wall_s']and row['psnr_db']==previous[case]['psnr_db']
    assert abs(sum(r['process_wall_s']for r in result['cases'])-outcome['five_case_sum_s'])<1e-6
    links=load(official/'immutable-video-links.json');assert links['count']==103 and links['ordinary_files']and links['original_outputs_unchanged']and links['no_GPU_rerun_or_timer_rewrite']
    assert links['selected_actual_timed_videos']+links['prior_accepted_SHA_equal_videos']==103
    public=load(ev/'public-repro-cpu-r1/summary.json');assert public['passed']and public['no_GPU_execution']and len(public['cases'])==3
    assert public['public_script_SHA256']==sha(root/'reproduce/run_nonv2_parallel.py')and all(r['source_count']==63 and r['exact_timed_manifest_and_assets']for r in public['cases'])
    print('Other nonV2 block63:three per-case sources/workloads/quality checks and fresh official103 adoption verified.')


if __name__=='__main__':verify()
