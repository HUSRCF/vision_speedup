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
    assert sha(root/'results/runtime.log')=='79edf05f4afd73df3ae236794324173a72e54c13c2168c3470456d6ba5d46bac'
    assert sha(root/'results/result.json')=='6a9ec891d56f625e31a9b24607b2d375babd1910a2dcec1eda9d84570888020b'
    result=json.loads((root/'results/result.json').read_text());assert result['status']=='accepted'
    cases=result['cases'];assert len(cases)==5 and sum(c['videos'] for c in cases)==103
    assert [c['rounds'] for c in cases]==[12,7,11,8,11]
    assert abs(sum(c['process_wall_s'] for c in cases)-850.836587)<1e-6
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
    assert outcome['V2_s']==312.462900 # historical213-source result
    import sys
    latest=root/'experiments/nccl-spatial-temporal-209'
    sys.path.insert(0,str(latest))
    try:
        from nccl_spatial_parallel209_sources import production_sources as spatial_sources
        new_files,new_proof=spatial_sources(code,'world-spatial-temporal')
    finally:sys.path.pop(0)
    assert new_proof==json.loads((latest/'timing-proof.json').read_text())
    new_manifest=json.loads((latest/'production-sources.json').read_text());assert len(new_manifest)==215
    for name,digest in new_manifest.items():
        actual=hashlib.sha256(new_files[name].encode()).hexdigest() if name in new_files else sha(code/name)
        assert actual==digest,name
    assets=json.loads((latest/'production-assets.json').read_text())
    assert hashlib.sha256(new_files['scripts/evaluation/world_model_interaction.py'].encode()).hexdigest()==assets['scripts/evaluation/world_model_interaction.py']
    assert sha(latest/'timing-proof.json')==assets['nccl-entry-timing-proof.json']
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert sha(code/name)==assets[name]
    accepted=json.loads((latest/'accepted-outcome.json').read_text())
    assert accepted['GPUs']==2 and accepted['source_count']==215 and accepted['all103_officially_validated']
    assert next(c['process_wall_s'] for c in cases if c['case'].endswith('stackbox_v2'))==accepted['V2_s']
    print('Verified: original209 base;historical213 and adopted215-source overlays;59 nonV2 sources;official103 log/JSON.')

if __name__=='__main__':main()
