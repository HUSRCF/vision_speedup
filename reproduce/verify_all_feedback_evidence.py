"""Check experimental provenance and official single-case evidence off GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def verify():
    root=Path(__file__).resolve().parents[1]
    here=root/'experiments/nonv2-all-feedback-vae'
    base=root/'experiments/nonv2-vae-all-display/unifolm'
    result=root/'results/aggressive-20261028/nonv2-all-feedback-vae'
    helper='src/unifolm_wma/utils/nonv2_nonfeedback_vae.py'
    spec=importlib.util.spec_from_file_location('all_feedback_builder',here/'build_nonv2_all_feedback_vae.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actual=(here/'production-overlay'/helper).read_text()
    assert actual==module.candidate_source((base/helper).read_text())
    old=json.loads((base/'nonv2-aggressive-production-sources.json').read_text())
    manifest=json.loads((here/'production-sources.json').read_text())
    assert set(old)==set(manifest) and len(manifest)==59
    assert {n for n in old if old[n]!=manifest[n]}=={helper}
    assert hashlib.sha256(actual.encode()).hexdigest()==manifest[helper]
    oldflags=json.loads((base.parent/'unitree_z1_stackbox-flags.json').read_text())
    flags=json.loads((here/'flags.json').read_text())
    assert flags==dict(oldflags,UNIFOLM_NONV2_VAE_SCOPE='all-feedback')
    for revision in (1,2):
        code=here/f'micro-r{revision}'
        for n,h in json.loads((code/'source-hashes.json').read_text()).items():
            assert hashlib.sha256((code/n).read_bytes()).hexdigest()==h
        report=json.loads((result/f'micro-r{revision}/summary.json').read_text())
        assert report['clean'] and report['passed'] and report['FP32_masters_input_RNG_flags_unchanged']
        assert report['all6_pairs_positive'] and len(report['pairs'])==6
        assert report['synthetic_latents'] and report['full_closed_loop_quality_unverified']
        if revision==2:
            assert report['cudnn_tf32'] and report['matmul_tf32'] and report['math_policy_matches_current_Z1']
    full=result/'full-r2'
    summary=json.loads((full/'summary.json').read_text())
    assert summary['clean'] and summary['workload_complete'] and summary['quality_passed']
    assert summary['official103_adoption_not_done'] and summary['official_default_unchanged']
    for name,h in summary['official_result_sha256'].items():
        # This unadopted experiment recorded the then-current221 aggregate.
        # Preserve and verify that exact historical pair after later adoption.
        historical=root/'results/history/official221-before-fused-half'/name
        assert hashlib.sha256(historical.read_bytes()).hexdigest()==h
    case=next(x for x in json.loads((full/'result.json').read_text())['cases'] if x['case']=='unitree_z1_stackbox')
    assert case['execution_ok'] and case['profile_complete'] and case['psnr_passed']
    assert (case['rounds'],case['videos'])==(12,25)
    assert case['process_wall_s']==summary['process_wall_s']==167.555672
    assert case['psnr_db']==summary['psnr_db']>25
    log=(full/'runtime.log').read_text()
    timing=re.findall(r'^TIMING case=(\S+) process_wall_s=([0-9.]+)$',log,re.M)
    assert timing==[('unitree_z1_stackbox',str(case['process_wall_s']))]
    records=re.findall(r'^NONV2_NONFEEDBACK_VAE (.+)$',log,re.M)
    assert len(records)==1
    decoder=json.loads(records[0])
    assert decoder['clips']==24 and not decoder['native_feedback_precision_and_flags_preserved']
    assert all(r['partitions']==[[False,0,16]] for r in decoder['records'])
    assert len([r for r in decoder['records'] if r.get('feedback_dtype')=='fp16'])==12
    for n,h in json.loads((here/'full-r2/asset-hashes.json').read_text()).items():
        assert hashlib.sha256((here/'full-r2'/n).read_bytes()).hexdigest()==h
    print('Historical unadopted all-feedback experiment and its exact221 aggregate evidence verified.')


if __name__=='__main__':verify()
