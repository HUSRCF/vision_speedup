"""Validate what the matched-stage Half screen actually measured."""
import hashlib,json
from pathlib import Path

def verify(root):
    p=root/'experiments/fp16-joined-residual-209/synthetic-r4';e=root/'results/aggressive-20261028/fp16-joined-residual-209/synthetic-r4'
    for n,h in json.loads((p/'source-hashes.json').read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
    base=json.loads((p/'baseline-production-identity.json').read_text());assert base['source_count']==len(base['manifest'])==209
    assert base['manifest']==json.loads((root/'unifolm/world-head-filter-cache-production-sources.json').read_text())
    summary=json.loads((e/'summary.json').read_text());r=json.loads((e/'comparison.json').read_text())
    assert summary['clean'] and summary['synthetic_only'] and not summary['official_adopted'] and not summary['model_or_video_quality_verified']
    assert summary['rows']==summary['precision_pass']==40 and summary['eligible']==summary['positive_medians']==0
    assert r['diagnostic_complete'] and r['all_dynamic_gates_passed'] and r['matched_pipeline_stages'] and r['actual_two_and_four_way_rows'] and len(r['rows'])==40 and r['dynamic_checks']==320
    configs=set()
    for x in r['rows']:
        assert x['baseline_stages']==x['stages'] and x['stages'] in (2,3) and x['precision_screen_passed'] and not x['eligible_for_real_operand_screen']
        assert len(x['cases'])==8 and x['RNG_unchanged'] and x['dynamic_restoration_signedzero_bits']
        assert all(c['finite'] and c['output_canaries'] and (not c['exact_required'] or c['signedzero_bits']) for c in x['cases'])
        assert len(x['timing']['pairs'])==6 and all(t['saved_ms']<0 for t in x['timing']['pairs'])
        configs.add((tuple(x['shape']),x['baseline_BM'],x['stages'],x['candidate_BM'],x['A_SHIFT']))
    assert len(configs)==40
    expected={((m,n,k),bm,s,cbm,shift) for m,n,k,bm in ((640,1280,5120,64),(320,1280,5120,64),(320,1280,5120,32),(160,1280,5120,32),(67,79,127,32)) for s in (2,3) for cbm in (32,64) for shift in (5,11)}
    assert configs==expected
    print('FP16_MATCHED_STAGE_SYNTHETIC passed=true;40 precision passes,240 slower timing pairs,no production adoption.')

if __name__=='__main__':verify(Path(__file__).resolve().parents[1])
