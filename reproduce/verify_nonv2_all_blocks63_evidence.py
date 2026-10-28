"""Validate the adopted full Z1 all-block run separately from adopted scores."""
import hashlib,json
from pathlib import Path


def verify():
    root=Path(__file__).resolve().parents[1]
    code=root/'experiments/nonv2-all-blocks59/full63-r1'
    evidence=root/'results/aggressive-20261028/nonv2-all-blocks59/full63-r1'
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=load(code/'production-sources.json');assert len(manifest)==63
    base=root/'experiments/nonv2-vae-all-display/unifolm'
    original=load(base/'nonv2-aggressive-production-sources.json');assert len(original)==59
    assert all(manifest[n]==h for n,h in original.items())
    for n,h in manifest.items():
        p=code/'production-overlay'/n
        assert sha(p if p.exists()else base/n)==h,n
    for n,h in load(code/'guard-source-hashes.json').items():assert sha(code/n)==h
    assets=load(code/'timing-asset-hashes.json')
    for n,h in assets.items():
        if n=='nccl-entry-timing-proof.json':p=code/'timing-proof.json'
        else:
            p=code/'production-overlay'/n
            if not p.exists():p=base/n
        assert sha(p)==h,n
    mode='src/unifolm_wma/utils/nonv2_ffn59_mode.py'
    for stage in ('decision','world'):
        assert sha(code/'production-overlay'/mode)==sha(root/f'experiments/nonv2-ffn-59/{stage}-model-r1/scripts/nonv2_ffn59_mode.py')
    assert '"cases": 36' in(code/'cpu-preflight.log').read_text()
    flags=load(code/'flags.json');assert flags['UNIFOLM_NCCL_PARALLEL']=='nonv2-all-blocks'
    original_flags=load(root/'experiments/nonv2-vae-all-display/unitree_z1_stackbox-flags.json')
    assert {k:v for k,v in flags.items()if k!='UNIFOLM_NCCL_PARALLEL'}==original_flags
    summary=load(evidence/'summary.json');assert summary['clean']and summary['workload_complete']and summary['rank_videos_exact']and not summary['official_adopted']
    case='unitree_z1_stackbox';row=next(r for r in load(evidence/'result.json')['cases']if r['case']==case)
    assert row['rounds']==12 and row['videos']==25 and row['execution_ok']and row['profile_complete']
    assert row['psnr_passed']==summary['quality_passed']and row['psnr_db']==summary['psnr_db']and row['process_wall_s']==summary['process_wall_s']
    assert load(evidence/'profiles/execution.json')[case]['exit_code']==0
    proof=load(evidence/'parallel/parallel-complete.json');assert proof['stage_calls']==dict(decision=600,world=600)and proof['rank_videos_SHA256_equal']and len(proof['video_hashes'])==25
    for rank in (0,1):
        controller=load(evidence/f'parallel/rank{rank}-controller.json');assert controller['stage_captures']==dict(decision=33,world=33)
        report=controller['controller'];assert report['captured']==66 and all(r['collectives']==1 for r in report['rows'])and report['temporal_block_captured']==34 and report['spatial_block_captured']==32
    verify_models(root)
    verify_adoption(root)
    print('Adopted nonV2 all-block63:756 parameter tests,33 shared gathers,full25-video Z1 and fresh official103 verified.')


def verify_models(root):
    for stage in ('decision','world'):
        source=root/f'experiments/nonv2-all-blocks59/{stage}-model-r1'
        for n,h in json.loads((source/'source-hashes.json').read_text()).items():
            assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h
        ev=root/f'results/aggressive-20261028/nonv2-all-blocks59/{stage}-model-r1'
        report=json.loads((ev/'summary.json').read_text());assert report['passed']and report['clean']and report['ready_for_model']
        for rank in report['ranks']:
            assert rank['partitioned_master_tensors']==756 and rank['temporal_inner_blocks']and rank['RNG_unchanged']and rank['baseline_restoration_bitwise']and rank['zero_FFN_changes_baseline']
            assert len(rank['cases'])==6 and all(e['gross_passed']for c in rank['cases']for e in c['errors'])
            assert len(rank['pairs'])==6 and all(p['saved_ms']>0 for p in rank['pairs'])
            ctl=rank['FFN_launches'];assert ctl['captured']==33 and ctl['temporal_block_captured']==17 and ctl['spatial_block_captured']==16 and ctl['same33_collectives_as_FFN_only']
    qualified=(root/'experiments/nonv2-all-blocks59/world-model-r1/scripts/nonv2_temporal_block59_mode.py').read_text()
    full=(root/'experiments/nonv2-all-blocks59/full63-r1/production-overlay/src/unifolm_wma/utils/nonv2_temporal_block59_mode.py').read_text()
    assert full==qualified.replace('import nonv2_ffn59_mode as base','from . import nonv2_ffn59_mode as base').replace('from nonv2_ffn59_mode import mode','from .nonv2_ffn59_mode import mode')


def verify_adoption(root):
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    outcome=load(root/'experiments/nonv2-all-blocks59/accepted-outcome.json')
    assert outcome['official_adopted']and outcome['all103_officially_validated']and outcome['source_count']==63 and outcome['GPUs']==2
    assert outcome['Z1_s']==152.203279 and outcome['V2_unchanged_s']==268.844229 and outcome['five_case_sum_s']==802.134091
    assert outcome['six_model_pairs_not_six_full_process_pairs']and outcome['rank25_video_SHA_equal']and outcome['all59_original_math_sources_unchanged']and outcome['original_process_timer_unchanged']
    source=root/'experiments/nonv2-all-blocks59/world-model-r1/scripts/nonv2_all_blocks59_mode.py'
    overlay=root/'experiments/nonv2-all-blocks59/full63-r1/production-overlay/src/unifolm_wma/utils/nonv2_all_blocks59_mode.py'
    assert overlay.read_text()==source.read_text().replace('import nonv2_ffn59_mode as base','from . import nonv2_ffn59_mode as base').replace('import nonv2_temporal_block59_mode as temporal','from . import nonv2_temporal_block59_mode as temporal')
    cpu=root/'results/aggressive-20261028/nonv2-all-blocks59/world-model-r1/check_nonv2_all_blocks59_cpu.py.log'
    assert 'cases=16 global_frame_head_mask=true whole_local_mix_tiles=true' in cpu.read_text()
    official=root/'results/aggressive-20261028/nonv2-all-blocks59/official103'
    for n,h in load(official/'script-hashes.json').items():assert sha(official/n)==h
    historical=root/'results/history/official-z1-blocks63-before-other63'
    for n,h in outcome['official_pair_sha256'].items():assert sha(official/n)==sha(historical/n)==h
    result=load(official/'result.json');assert result==load(historical/'result.json')and result['status']=='accepted'and result['psnr_validation']['video_count']==103
    case=next(r for r in result['cases']if r['case']=='unitree_z1_stackbox')
    assert case['process_wall_s']==outcome['Z1_s']and case['psnr_db']==outcome['PSNR_dB']and case['psnr_passed']
    current=next(r for r in load(root/'results/result.json')['cases']if r['case']=='unitree_z1_stackbox')
    assert current['process_wall_s']==case['process_wall_s']and current['psnr_db']==case['psnr_db']and current['psnr_passed']
    links=load(official/'immutable-video-links.json');assert links['count']==103 and all(links[k]for k in('ordinary_files','original_outputs_unchanged','selected25_from_new_timed_run','other78_SHA_equal_prior_accepted','no_GPU_rerun_or_timer_rewrite'))
    before=root/'results/history/official222-z1-fused-before-blocks63'
    for n,h in outcome['previous_pair_sha256'].items():assert sha(before/n)==h
    previous=load(before/'result.json');assert previous['status']=='accepted'
    assert abs(sum(r['process_wall_s']for r in previous['cases'])-outcome['five_case_sum_s']-outcome['saved_vs_historical_Z1_s'])<1e-6
    entry=(root/'reproduce/run_z1_parallel.py').read_text();assert "default='all63'"in entry
    public=load(root/'results/aggressive-20261028/nonv2-all-blocks59/public-repro-cpu-r1/summary.json')
    assert public['passed']and public['no_GPU_execution']and public['source_count']==63 and public['exact_timed_manifest_and_assets']
    assert public['public_script_SHA256']==sha(root/'reproduce/run_z1_parallel.py')and public['home_free_before']==public['home_free_after']


if __name__=='__main__':verify()
