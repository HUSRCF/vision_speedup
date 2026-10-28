"""Verify exact current222 diagnostic sources and aggregate-only GPU profiles."""
import hashlib,json,statistics
from pathlib import Path


def verify():
    root=Path(__file__).resolve().parents[1]
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    original=load(root/'experiments/initial-boundary-221/production222/production-sources.json');assert len(original)==222
    expected={'decision':dict(spatial_captured=16,ffn_captured=0,temporal_captured=0,native_attention=70,resblock_grid=22,temporal_block=0,initial_FFN=0,initial_block=0),
              'world':dict(spatial_captured=32,ffn_captured=16,temporal_captured=88,native_attention=140,resblock_grid=22,temporal_block=16,initial_FFN=1,initial_block=1)}
    for stage in ['decision','world']:
        code=root/f'experiments/adopted222-profile/{stage}-r1';ev=root/f'results/aggressive-20261028/adopted222-profile/{stage}-r1'
        for n,h in load(code/'source-hashes.json').items():assert sha(code/n)==h
        identity=load(code/'production-identity.json');assert identity['source_count']==222 and identity['manifest']==original
        summary=load(ev/'summary.json');assert summary['clean']and summary['passed']and summary['profile_only']and summary['no_official_score']and not summary['arithmetic_or_routing_changed']and not summary['raw_trace_or_addresses_exported']
        telemetry=[json.loads(line)for line in(ev/'gpu-telemetry.jsonl').read_text().splitlines()];assert len(telemetry)==summary['observations']and telemetry
        for rank in (0,1):
            row=load(ev/f'rank{rank}-profile.json');assert row['profile_only']and row['stage']==stage and row['baseline_source_count']==222 and not row['arithmetic_or_routing_changed']
            assert row['no_official_score']and row['full_video_PSNR_unverified']and row['aggregate_only_no_raw_trace_or_addresses']and row['profiled_durations_include_profiler_perturbation']
            assert row['controller_counts']==expected[stage]
            assert len(row['replay_checks'])==6 and all(c['three_outputs_signedzero_bitwise']for c in row['replay_checks'])and row['RNG_unchanged']
            groups=row['unprofiled_timing_groups'];assert len(groups)==6 and all(len(g['samples_ms'])==7 and g['median_ms']==statistics.median(g['samples_ms'])for g in groups)
            assert row['unprofiled_median_ms']==statistics.median(x for g in groups for x in g['samples_ms'])
            replays=row['profiler_replays'];assert replays==5
            rows=row['kernel_rows'];assert rows and sum(r['count']for r in rows)==row['CUDA_event_count']
            for k in rows:
                assert k['count']>0 and k['count']%replays==0 and k['total_us']>=0
                assert abs(k['per_forward_ms']-k['total_us']/replays/1000)<1e-8
                assert abs(k['mean_us']-k['total_us']/k['count'])<1e-8
            assert abs(sum(k['per_forward_ms']for k in rows)-row['kernel_sum_per_forward_ms'])<1e-8
    print('Adopted222 profiles:exact sources,clean two-rank replay checks and scalar-only kernel evidence verified;no new score.')

if __name__=='__main__':verify()
