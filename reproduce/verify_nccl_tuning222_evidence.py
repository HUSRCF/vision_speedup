"""Check communication-only provenance separately from full inference results."""
import hashlib
import json
from pathlib import Path


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify():
    root=Path(__file__).resolve().parents[1];code=root/'experiments/nccl-tuning222';ev=root/'results/aggressive-20261028/nccl-tuning222'
    reference=root/'results/aggressive-20261028/initial-boundary-221/private-tmpfs-full/full-candidate-r1/parallel/rank0-controller.json'
    for scope,count in [('micro',8),('combos',5)]:
        source=code/(scope+'-r1');out=ev/(scope+'-r1')
        for n,h in load(source/'source-hashes.json').items():assert sha(source/n)==h
        payload=load(source/'payloads.json');assert payload['source_controller_sha256']==sha(reference)and payload['source_count']==222
        assert payload['FP32_local_bytes']and payload['phase_order_is_sorted_histogram_not_actual_model_order']
        for stage,calls,MiB in [('decision',38,450),('world',121,1399.375)]:
            hist=payload['stages'][stage];assert sum(hist.values())==calls and sum(int(n)*c for n,c in hist.items())/2**20==MiB
        configs=load(source/'configurations.json');summary=load(out/'sweep-summary.json');assert len(configs)==count and summary['clean']and summary['all_configurations_completed']and summary['no_model_or_video_quality_claim']
        for config,measured in zip(configs,summary['results']):
            assert config==measured['config'];folder=out/config['name'];report=load(folder/'nccl-geglu-model-report.json');assert report['clean']and report['passed']and len(report['ranks'])==2
            for rank in report['ranks']:
                assert rank['all_gather_exact']and rank['graph_exact']and rank['no_model_or_video_quality_claim']and len(rank['records'])==5
                assert rank['config']==config and rank['NCCL_version']==[2,20,5]
                for row in rank['records']:
                    assert row['exact_copy']and row['graph_collectives']==32 and len(row['states'])==4 and all(c['bitwise']for c in row['states'])
                    groups=row['amortized_graph_timing']['groups'];assert len(groups)==6 and all(len(g['samples_ms'])==5 for g in groups)
                for stage in('decision','world'):
                    assert rank['phases'][stage]['size_histogram_surrogate_only']and rank['phases'][stage]['not_actual_model_interleaving_or_inference_score']
            for stage in('decision','world'):assert measured['phase_ms'][stage]==max(r['phases'][stage]['timing']['median_ms']for r in report['ranks'])
    combo=load(ev/'combos-r1/sweep-summary.json')['results'];chosen=next(r for r in combo if r['config']['name']=='channels32_write')
    assert all(sum(chosen['phase_ms'].values())<sum(r['phase_ms'].values())for r in combo if r['config']['name'].startswith('default'))
    transport=load(ev/'combos-r1/channels32_write/transport-summary.json');assert any(s.startswith('32 coll channels')for s in transport['channel_descriptions'])and transport['transport_counts'].get('P2P/CUMEM',0)>0 and transport['transport_counts'].get('P2P/CUMEM/read',0)==0
    print('NCCL tuning:13 clean communication screens, copy equality and actual222 payload counts verified;not an inference score.')


if __name__=='__main__':verify()
