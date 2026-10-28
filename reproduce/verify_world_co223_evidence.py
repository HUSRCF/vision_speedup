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
    print('CO223 qualification:14 geometries/two layouts/two ranks/six states,9 selected owners and six positive full-model pairs verified; no formal score inferred.')


if __name__=='__main__':verify()
