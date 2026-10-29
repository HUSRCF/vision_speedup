"""Check exact226 overlays and the scope of four-rank model evidence."""
import hashlib,json,statistics
from pathlib import Path

def verify(root):
    for stage in ('decision','world'):
        p=root/f'experiments/four-way226/{stage}-model-r1';e=root/f'results/aggressive-20261028/four-way226/{stage}-model-r1'
        manifest=json.loads((p/'production-sources.json').read_text());base=json.loads((p/'baseline-production-identity.json').read_text())['manifest'];assert len(manifest)==226 and len(base)==225
        actual=dict(base)
        for f in (p/'production-overlay').rglob('*.py'):actual[str(f.relative_to(p/'production-overlay'))]=hashlib.sha256(f.read_bytes()).hexdigest()
        assert actual==manifest
        for n,h in json.loads((p/'source-hashes.json').read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
        assert 'cases=40' in (p/'cpu-preflight.log').read_text()
        s=json.loads((e/'summary.json').read_text());assert s['passed'] and s['clean'] and s['GPUs']==4 and s['ready_for_model']
        assert s['baseline_math_original209_before_controller_install'] and s['paired_comparator_is_not_adopted225_two_GPU'] and not s['full_V2_process_or_video_quality_qualification'] and s['no_official_score']
        assert len(s['ranks'])==4
        for r in s['ranks']:
            assert r['stage']==stage and r['RNG_unchanged'] and r['baseline_restoration_bitwise'] and r['all_diffusion_all_masters_tested']
            assert r['Linear_master_tensors']==570 and r['norm_master_tensors']==782 and r['Conv2d_master_tensors']==568
            assert len(r['cases'])==6 and all(e['signedzero_bits'] for c in r['cases'] for e in c['errors'])
            assert len(r['pairs'])==6 and all(x['saved_ms']>0 for x in r['pairs'])
            assert r['candidate_source_count']==226 and r['actual_GPUs']==4
            assert r['attention_launches']['world_GPUs']==4
        print('FOUR_WAY226_MODEL_EVIDENCE',stage,'passed=true','candidate_ms='+str([statistics.median(x['candidate_ms'] for x in r['pairs']) for r in s['ranks']]))
    for stage in ('decision','world'):
        p=root/f'experiments/adopted225-profile/{stage}-r1';e=root/f'results/aggressive-20261028/adopted225-profile/{stage}-r1'
        for n,h in json.loads((p/'source-hashes.json').read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
        identity=json.loads((p/'production-identity.json').read_text());assert identity['source_count']==len(identity['manifest'])==225
        summary=json.loads((e/'summary.json').read_text());assert summary['clean'] and summary['passed'] and summary['no_official_score'] and summary['profile_only']
        for rank in (0,1):
            r=json.loads((e/f'rank{rank}-profile.json').read_text());assert r['stage']==stage and r['baseline_source_count']==225 and r['RNG_unchanged'] and r['aggregate_only_no_raw_trace_or_addresses']
            assert len(r['replay_checks'])==6
    print('ADOPTED225_PROFILE_EVIDENCE passed=true')
    full=[]
    for revision in (1,2):
        p=root/f'experiments/four-way226/full-candidate-r{revision}'
        manifest=json.loads((p/'production-sources.json').read_text())
        base=json.loads((root/'experiments/four-way226/world-model-r1/baseline-production-identity.json').read_text())['manifest'];actual=dict(base)
        for f in (p/'production-overlay').rglob('*.py'):actual[str(f.relative_to(p/'production-overlay'))]=hashlib.sha256(f.read_bytes()).hexdigest()
        assert actual==manifest and len(manifest)==226;full.append(manifest)
        for n,h in json.loads((p/'guard-source-hashes.json').read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
        assets=json.loads((p/'timing-asset-hashes.json').read_text());assert hashlib.sha256((p/'timing-proof.json').read_bytes()).hexdigest()==assets['nccl-entry-timing-proof.json']
        assert 'children' in (p/'check_four_way226_lifecycle_cpu.py.log').read_text() and 'cases=40' in (p/'check_four_way226_cpu.py.log').read_text()
    assert full[0]==full[1]  # Guard repair does not alter production arithmetic.
    e=root/'results/aggressive-20261028/four-way226/full-candidate-r1'
    summary=json.loads((e/'summary.json').read_text());assert not summary['clean'] and not summary['official_adopted'] and summary['workload_complete'] and summary['quality_passed'] and summary['rank_videos_exact'] and summary['baseline_videos_exact']
    assert summary['process_wall_s']==260.270918 and 'guard' in summary['failure_reason']
    result=json.loads((e/'result.json').read_text());row=next(r for r in result['cases'] if r['case']=='unitree_z1_dual_arm_stackbox_v2')
    assert row['rounds']==11 and row['videos']==23 and row['psnr_passed'] and row['execution_ok'] and row['profile_complete'] and row['process_wall_s']==260.270918
    proof=json.loads((e/'parallel/parallel-complete.json').read_text());assert proof['GPUs']==4 and proof['helper_exit_codes']==[0,0,0] and proof['rank_videos_SHA256_equal'] and proof['stage_calls']==dict(decision=550,world=550)
    for rank in range(4):
        ctl=json.loads((e/f'parallel/rank{rank}-controller.json').read_text());assert ctl['stage_captures']==dict(decision=16,world=32)
    b=json.loads((e/'baseline-video-comparison.json').read_text());assert b['all23_SHA256_equal'] and len(b['current_hashes'])==23 and b['current_hashes']==b['baseline_hashes']
    cpu=json.loads((root/'results/aggressive-20261028/four-way226/public-repro-cpu-r1/summary.json').read_text());assert cpu['passed'] and cpu['source_count']==226 and cpu['no_GPU_execution'] and cpu['exact_timed_manifest_and_assets']
    assert hashlib.sha256((root/'reproduce/run_v2_four_way226.py').read_bytes()).hexdigest()==cpu['public_script_SHA256']
    print('FOUR_WAY226_FULL_PREPARATION passed=true; first260.270918 wrapper-guard failure retained, not adopted.')

if __name__=='__main__':verify(Path(__file__).resolve().parents[1])
