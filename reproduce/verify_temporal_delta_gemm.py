"""Verify the primitive's scope, full refresh exactness and complete pair cost."""
import hashlib,json,statistics
from pathlib import Path

def verify(root):
    for revision in (1,2,3):
        p=root/f'experiments/temporal-delta-gemm/primitive-r{revision}';e=root/f'results/aggressive-20261029/temporal-delta-gemm/primitive-r{revision}'
        if not p.exists():continue
        for n,h in json.loads((p/'source-hashes.json').read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
        identity=json.loads((p/'baseline-identity.json').read_text());base=root/'unifolm/world-head-filter-cache-production-sources.json'
        assert len(identity['manifest'])==identity['source_count']==209 and identity['manifest']==json.loads(base.read_text())and identity['manifest_sha256']==hashlib.sha256(base.read_bytes()).hexdigest()
        summary=json.loads((e/'summary.json').read_text());report=json.loads((e/'comparison.json').read_text())
        assert summary['clean']and summary['synthetic_only']and not summary['full_model_and_video_quality_verified']and not summary['official_adopted']
        assert len(report['rows'])==6 and report['all_pack_cache_and_refresh_costs_timed']and report['full_refresh_original_BK64_bitwise']
        assert {(tuple(x['shape']),x['weight_low_branch'])for x in report['rows']}=={(s,b)for s in ((2560,1280,5120),(640,1280,5120),(67,79,127))for b in (False,True)}
        for row in report['rows']:
            assert row['RNG_unchanged']and row['all_pack_and_cache_costs_timed']and len(row['states'])==8 and row['automatic_master_owner_not_implemented']and not row['full_model_and_video_quality_verified']
            for state in row['states']:
                assert state['finite']and state['relative_l2']<.001
                if state['exact_required']:assert state['signedzero_bits']
            pairs=row['timing']['pairs'];assert len(pairs)==6
            assert row['timing']['all6_faster']==all(x['saved_two_steps_ms']>0 for x in pairs)
            assert row['timing']['median_saved_two_steps_ms']==statistics.median(x['saved_two_steps_ms']for x in pairs)
            if revision==3:
                assert len(row['live_flag_graph_checks'])==3
                for state in row['live_flag_graph_checks']:
                    assert state['finite']and state['relative_l2']<.001
                    if state['exact_required']:assert state['signedzero_bits']
        assert summary['fast_rows']==sum(x['timing']['all6_faster']for x in report['rows'])
        if revision>=2:assert report['device_predicated_single_graph']and report['per_operator_flag_fill_cost_included']
        print('TEMPORAL_DELTA_PRIMITIVE r'+str(revision)+' passed=true; full refresh exact, delta approximate, all pack/cache cost counted, no model/video claim.')

if __name__=='__main__':verify(Path(__file__).resolve().parents[1])
