"""Verify fixed production source and historical official-result evidence."""
import hashlib,json,re
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    root=Path(__file__).resolve().parents[1];code=root/'unifolm'
    manifest=json.loads((code/'world-head-filter-cache-production-sources.json').read_text())
    assert len(manifest)==209
    assert set(manifest)=={str(p.relative_to(code)) for p in (code/'src').rglob('*.py')}
    for name,digest in manifest.items():assert sha(code/name)==digest,name
    assert sha(code/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so')=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    assert sha(root/'results/runtime.log')=='65030f4108d00766aacba449d0f5de320170119c5ae232ca75db8fca48c352ca'
    assert sha(root/'results/result.json')=='8a4d4807408bca2e5533c040cb67279154bc305f0f179f5bec6ea0d2276c0b38'
    result=json.loads((root/'results/result.json').read_text());assert result['status']=='accepted'
    cases=result['cases'];assert len(cases)==5 and sum(c['videos'] for c in cases)==103
    assert [c['rounds'] for c in cases]==[12,7,11,8,11]
    assert abs(sum(c['process_wall_s'] for c in cases)-861.368824)<1e-6
    log=(root/'results/runtime.log').read_text()
    timings=re.findall(r'^TIMING case=(\S+) process_wall_s=([0-9.]+)$',log,re.M)
    assert len(timings)==5 and len({case for case,_ in timings})==5
    assert {c['case']:c['process_wall_s'] for c in cases}=={case:float(value) for case,value in timings}
    assert all(c['psnr_passed'] and c['execution_ok'] and c['profile_complete'] for c in cases)
    nonv2=root/'experiments/nonv2-vae-all-display'
    variant=nonv2/'unifolm';other=json.loads((variant/'nonv2-aggressive-production-sources.json').read_text())
    assert len(other)==59
    for name,digest in other.items():assert sha(variant/name)==digest,name
    for name,digest in json.loads((nonv2/'asset-sha256.json').read_text()).items():assert sha(variant/name)==digest,name
    accepted=json.loads((nonv2/'accepted-cases.json').read_text())
    for row in cases:
        if row['case'] in accepted:assert row['process_wall_s']==accepted[row['case']]['process_wall_s']
    import importlib.util
    overlay=root/'experiments/nccl-ffn-209/timed-entry'
    spec=importlib.util.spec_from_file_location('parallel_sources',overlay/'nccl_parallel209_sources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files,proof=module.production_sources(code,'world-ffn-temporal')
    assert proof==json.loads((overlay/'timing-proof.json').read_text())
    parallel=json.loads((overlay/'production-sources.json').read_text());assert len(parallel)==213
    for name,digest in parallel.items():
        actual=hashlib.sha256(files[name].encode()).hexdigest() if name in files else sha(code/name)
        assert actual==digest,name
    assets=json.loads((overlay/'production-assets.json').read_text())
    assert hashlib.sha256(files['scripts/evaluation/world_model_interaction.py'].encode()).hexdigest()==assets['scripts/evaluation/world_model_interaction.py']
    assert sha(overlay/'timing-proof.json')==assets['nccl-entry-timing-proof.json']
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert sha(code/name)==assets[name]
    outcome=json.loads((overlay/'accepted-outcome.json').read_text());assert outcome['GPUs']==2
    assert next(c['process_wall_s'] for c in cases if c['case'].endswith('stackbox_v2'))==outcome['V2_s']
    print('Verified: original209 base plus exact213-source dualGPU overlay;59 nonV2 sources;official103 log/JSON.')

if __name__=='__main__':main()
