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
    full={}
    for variant in ('candidate','baseline'):
        source=code/f'full-{variant}-r1';out=ev/f'full-{variant}-r1'
        for n,h in load(source/'source-hashes.json').items():assert sha(source/n)==h
        manifest=load(source/'production-sources.json');assert len(manifest)==222
        summary=load(out/'summary.json');assert summary['verified_source_count']==222 and summary['same_source_as_adopted222']and summary['original_evaluator_timer_unchanged']
        assert all(summary[k]for k in ('clean','quality_passed','workload_complete','baseline_videos_exact','rank_videos_exact'))
        assert summary['gpu_candidates']==['0','1']
        selected=next(r for r in load(out/'result.json')['cases']if r['case']=='unitree_z1_dual_arm_stackbox_v2')
        assert selected['rounds']==11 and selected['videos']==23 and selected['execution_ok']and selected['profile_complete']and selected['psnr_passed']
        assert selected['process_wall_s']==summary['process_wall_s']and selected['psnr_db']==summary['psnr_db']
        assert load(out/'profiles/execution.json')['unitree_z1_dual_arm_stackbox_v2']['exit_code']==0
        comparison=load(out/'baseline-video-comparison.json');assert comparison['all23_SHA256_equal']and comparison['current_hashes']==comparison['baseline_hashes']and len(comparison['current_hashes'])==23
        proof=load(out/'parallel/parallel-complete.json');assert proof['rank_videos_SHA256_equal']and proof['stage_calls']==dict(decision=550,world=550)
        profile=load(out/'profiles/unitree_z1_dual_arm_stackbox_v2.json');assert profile['total_process_wall_s']==summary['process_wall_s']
        full[variant]=(manifest,load(source/'timing-asset-hashes.json'),load(out/'flags.json'),summary)
    candidate,baseline=full['candidate'],full['baseline'];assert candidate[:2]==baseline[:2]
    changes={k:v for k,v in candidate[2].items()if baseline[2].get(k)!=v}
    assert changes==dict(NCCL_MIN_NCHANNELS='32',NCCL_MAX_NCHANNELS='32',NCCL_P2P_READ_ENABLE='0')
    assert set(candidate[2])-set(baseline[2])==set(changes)
    actual=load(ev/'full-candidate-r1/effective-tuning-proof.json');assert actual['passed']and actual['observed_live_rank_environment']and {r['rank']for r in actual['ranks']}=={0,1}
    assert all(r['actual_tuning_env']==changes for r in actual['ranks'])
    assert abs(baseline[3]['process_wall_s']-candidate[3]['process_wall_s']-.431037)<1e-9
    print('NCCL tuning:13 communication screens and one same-source222 complete V2 pair verified; official pair unchanged.')


if __name__=='__main__':verify()
