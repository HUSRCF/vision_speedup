"""Check immutable feature-reuse sources and the scope of GPU sanity evidence."""
import hashlib,json
from pathlib import Path

def verify(root):
    p=root/'experiments/crossstep211';e=root/'results/aggressive-20261029/crossstep211'
    base=json.loads((root/'unifolm/world-head-filter-cache-production-sources.json').read_text())
    manifest=json.loads((p/'production-sources.json').read_text());assert len(base)==209 and len(manifest)==211
    changed={n for n,h in manifest.items()if base.get(n)!=h}
    assert changed=={'src/unifolm_wma/utils/'+n for n in ('cuda_graph.py','crossstep_feature_reuse.py','crossstep_feature_runtime.py')}
    for n,h in manifest.items():
        q=p/'production-overlay'/n if n in changed else root/'unifolm'/n
        assert hashlib.sha256(q.read_bytes()).hexdigest()==h,n
    old=(root/'unifolm/src/unifolm_wma/utils/cuda_graph.py').read_text()
    new=(p/'production-overlay/src/unifolm_wma/utils/cuda_graph.py').read_text()
    assert new.replace('pool=getattr(self, "experimental_capture_pool", None),','')==old or new.replace(', pool=getattr(self, "experimental_capture_pool", None)','')==old
    for n,h in json.loads((p/'sanity-source-hashes.json').read_text()).items():
        assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
    full=json.loads((p/'full-source-hashes.json').read_text())
    for n,h in full['scripts'].items():assert hashlib.sha256((p/('full-'+n)).read_bytes()).hexdigest()==h
    for n,h in full['assets'].items():assert hashlib.sha256((p/'timing-assets'/n).read_bytes()).hexdigest()==h
    r=json.loads((e/'sanity-comparison.json').read_text())
    assert r['diagnostic_complete'] and r['all_dynamic_gates_passed'] and r['synthetic_only'] and not r['real_WMA_or_video_quality_verified']
    assert r['three_outputs_checked_calls']==200 and r['graph_count']==3 and r['RNG_unchanged'] and r['world_refresh_and_reuse_share_graph_pool']
    assert r['counts']['decision_calls']==r['counts']['world_calls']==100 and r['counts']['world_reuse']==28
    assert r['invalidations']['master_version']==r['invalidations']['condition_change']==1
    assert len(r['completed_generations'])==4 and all(g['steps']==50 for g in r['completed_generations'])
    assert not r['sampler_step_count_reduced'] and not r['round_count_reduced']
    full=e/'full-r1'
    if (full/'summary.json').exists():
        summary=json.loads((full/'summary.json').read_text());result=json.loads((full/'result.json').read_text())
        row=next(x for x in result['cases']if x['case']=='unitree_z1_dual_arm_stackbox_v2')
        assert summary['clean'] and summary['workload_complete'] and not summary['official_adopted']
        assert row['rounds']==11 and row['videos']==23 and row['execution_ok'] and row['profile_complete']
        assert row['process_wall_s']==summary['process_wall_s'] and row['psnr_db']==summary['psnr_db'] and row['psnr_passed']==summary['quality_passed']
        report=json.loads((full/'profiles/unitree_z1_dual_arm_stackbox_v2.crossstep.json').read_text())
        assert report['counts']['decision_calls']==report['counts']['world_calls']==550 and report['counts']['world_reuse']==165
        assert len(report['completed_generations'])==22 and all(g['steps']==50 for g in report['completed_generations'])
    for revision in (3,4,5,6,7,8):
        q=p/f'full-r{revision}'
        if not q.exists():continue
        identity=json.loads((q/'identity.json').read_text());assert len(identity['manifest'])==211 and not identity['official_adopted']
        for n,h in identity['manifest'].items():
            source=q/'production-overlay'/n if base.get(n)!=h else root/'unifolm'/n
            assert hashlib.sha256(source.read_bytes()).hexdigest()==h,n
        for n,h in identity['scripts'].items():assert hashlib.sha256((q/n).read_bytes()).hexdigest()==h,n
        for n,h in identity['assets'].items():assert hashlib.sha256((q/'timing-assets'/n).read_bytes()).hexdigest()==h,n
        full=e/f'full-r{revision}'
        if not(full/'summary.json').exists():continue
        summary=json.loads((full/'summary.json').read_text());result=json.loads((full/'result.json').read_text())
        row=next(x for x in result['cases']if x['case']=='unitree_z1_dual_arm_stackbox_v2')
        assert summary['clean'] and summary['workload_complete'] and not summary['official_adopted']
        assert row['rounds']==11 and row['videos']==23 and row['execution_ok'] and row['profile_complete']
        assert row['process_wall_s']==summary['process_wall_s'] and row['psnr_db']==summary['psnr_db'] and row['psnr_passed']==summary['quality_passed']
        report=json.loads((full/'profiles/unitree_z1_dual_arm_stackbox_v2.crossstep.json').read_text())
        if identity['policy'].get('scope')=='correction-observe':
            assert summary['diagnostic_only']and summary['not_a_speedup_candidate']and summary['baseline_videos_exact']
            assert report['inference_outputs_unmodified']and report['no_native_argument_inspection']and report['counts']==dict(decision=550,world=550)and len(report['rows'])==50
            hashes=json.loads((full/'baseline-video-comparison.json').read_text());assert hashes['all23_SHA256_equal']and len(hashes['current_hashes'])==23 and hashes['current_hashes']==hashes['baseline_hashes']
            continue
        first,last=identity['policy']['start'],identity['policy']['stop'];expected_reuse=11*sum(first<=k<last and(k-first)%2!=0 for k in range(50))
        assert report['counts']['decision_calls']==report['counts']['world_calls']==550 and report['counts']['world_reuse']==expected_reuse
        assert len(report['completed_generations'])==22 and all(g['steps']==50 for g in report['completed_generations'])
        if identity['policy'].get('scope')=='spatial':
            assert report['all_convolutions_and_heads_live']and report['reuse_graph_head_filter_cache']and len(report['selected_blocks'])==16
        else:assert report['reuse_graph_native_head_transforms_live'] and not report['reuse_graph_head_filter_cache']
    q=p/'residual-sanity'
    if q.exists():
        identity=json.loads((q/'identity.json').read_text());assert len(identity['manifest'])==211
        for n,h in identity['manifest'].items():
            source=q/'production-overlay'/n if base.get(n)!=h else root/'unifolm'/n
            assert hashlib.sha256(source.read_bytes()).hexdigest()==h
        for n,h in identity['scripts'].items():assert hashlib.sha256((q/n).read_bytes()).hexdigest()==h
        r=json.loads((e/'residual-sanity/comparison.json').read_text())
        assert r['boundary_residual']and r['current_boundary_input_live']and r['three_outputs_checked_calls']==200 and r['RNG_unchanged']
        assert r['graph_count']==3 and r['synthetic_only']and not r['real_WMA_or_video_quality_verified']
    q=p/'spatial-sanity'
    if q.exists():
        identity=json.loads((q/'identity.json').read_text());assert len(identity['manifest'])==211
        for n,h in identity['manifest'].items():
            source=q/'production-overlay'/n if base.get(n)!=h else root/'unifolm'/n
            assert hashlib.sha256(source.read_bytes()).hexdigest()==h
        for n,h in identity['scripts'].items():assert hashlib.sha256((q/n).read_bytes()).hexdigest()==h
        r=json.loads((e/'spatial-sanity/comparison.json').read_text())
        assert r['scope']=='spatial'and r['all_convolutions_and_heads_live']and r['three_outputs_checked_calls']==200 and r['RNG_unchanged']
        assert r['graph_count']==3 and r['synthetic_only']and not r['real_WMA_or_video_quality_verified']
    print('CROSSSTEP211 passed=true; synthetic graph consistency is separate from full-video PSNR; no official adoption.')

if __name__=='__main__':verify(Path(__file__).resolve().parents[1])
