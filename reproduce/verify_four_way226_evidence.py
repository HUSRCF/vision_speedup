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

if __name__=='__main__':verify(Path(__file__).resolve().parents[1])
