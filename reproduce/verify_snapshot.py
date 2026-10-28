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
    assert sha(root/'results/runtime.log')=='73ad107f5741bfe07a38f00c4777ea0e2ccc28b0d79e4d8321525397dd6579b5'
    assert sha(root/'results/result.json')=='3d358d1b575bb6b23d55fd36cde8725f055ac27811ef994101d45b77921a69e7'
    result=json.loads((root/'results/result.json').read_text());assert result['status']=='accepted'
    cases=result['cases'];assert len(cases)==5 and sum(c['videos'] for c in cases)==103
    assert [c['rounds'] for c in cases]==[12,7,11,8,11]
    assert abs(sum(c['process_wall_s'] for c in cases)-889.570845)<1e-6
    log=(root/'results/runtime.log').read_text()
    timings=re.findall(r'^TIMING case=(\S+) process_wall_s=([0-9.]+)$',log,re.M)
    assert len(timings)==5 and len({case for case,_ in timings})==5
    assert {c['case']:c['process_wall_s'] for c in cases}=={case:float(value) for case,value in timings}
    assert all(c['psnr_passed'] and c['execution_ok'] and c['profile_complete'] for c in cases)
    print('Verified: 209 production sources; pinned extension; original log/JSON; five timings; 49 rounds / 103 videos.')

if __name__=='__main__':main()
