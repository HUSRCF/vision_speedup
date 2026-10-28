"""Validate the experimental full Z1 FFN run separately from adopted scores."""
import hashlib,json
from pathlib import Path


def verify():
    root=Path(__file__).resolve().parents[1]
    code=root/'experiments/nonv2-ffn-59/full61-r1'
    evidence=root/'results/aggressive-20261028/nonv2-ffn-59/full61-r1'
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=load(code/'production-sources.json');assert len(manifest)==61
    base=root/'experiments/nonv2-vae-all-display/unifolm'
    original=load(base/'nonv2-aggressive-production-sources.json');assert len(original)==59
    assert all(manifest[n]==h for n,h in original.items())
    for n,h in manifest.items():
        p=code/'production-overlay'/n
        assert sha(p if p.exists()else base/n)==h,n
    for n,h in load(code/'guard-source-hashes.json').items():assert sha(code/n)==h
    assets=load(code/'timing-asset-hashes.json')
    for n,h in assets.items():
        if n=='nccl-entry-timing-proof.json':p=code/'timing-proof.json'
        else:
            p=code/'production-overlay'/n
            if not p.exists():p=base/n
        assert sha(p)==h,n
    mode='src/unifolm_wma/utils/nonv2_ffn59_mode.py'
    for stage in ('decision','world'):
        assert sha(code/'production-overlay'/mode)==sha(root/f'experiments/nonv2-ffn-59/{stage}-model-r1/scripts/nonv2_ffn59_mode.py')
    assert '"cases": 36' in(code/'cpu-preflight.log').read_text()
    flags=load(code/'flags.json');assert flags['UNIFOLM_NCCL_PARALLEL']=='nonv2-ffn'
    original_flags=load(root/'experiments/nonv2-vae-all-display/unitree_z1_stackbox-flags.json')
    assert {k:v for k,v in flags.items()if k!='UNIFOLM_NCCL_PARALLEL'}==original_flags
    summary=load(evidence/'summary.json');assert summary['clean']and summary['workload_complete']and summary['rank_videos_exact']and not summary['official_adopted']
    case='unitree_z1_stackbox';row=next(r for r in load(evidence/'result.json')['cases']if r['case']==case)
    assert row['rounds']==12 and row['videos']==25 and row['execution_ok']and row['profile_complete']
    assert row['psnr_passed']==summary['quality_passed']and row['psnr_db']==summary['psnr_db']and row['process_wall_s']==summary['process_wall_s']
    assert load(evidence/'profiles/execution.json')[case]['exit_code']==0
    proof=load(evidence/'parallel/parallel-complete.json');assert proof['stage_calls']==dict(decision=600,world=600)and proof['rank_videos_SHA256_equal']and len(proof['video_hashes'])==25
    for rank in (0,1):
        controller=load(evidence/f'parallel/rank{rank}-controller.json');assert controller['stage_captures']==dict(decision=33,world=33)
        report=controller['controller'];assert report['captured']==66 and all(r['collectives']==1 for r in report['rows'])
    print('Experimental nonV2 FFN61: original59 math, timed entry, full12 rounds/25 videos and selected official quality verified; adoption remains separate.')


if __name__=='__main__':verify()
