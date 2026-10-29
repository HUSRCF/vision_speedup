"""Check complete per-case AB evidence and unchanged inference mathematics."""
import hashlib,json
from pathlib import Path


def verify():
    repo=Path(__file__).resolve().parents[1]
    code=repo/'experiments/nonv2-root-transport64'
    evidence=repo/'results/aggressive-20261029/nonv2-root-transport64'
    base=repo/'experiments/nonv2-vae-all-display/unifolm'
    load=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    runtime='src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'
    mover='src/unifolm_wma/utils/nccl_root_model_move.py'
    qualified=repo/'experiments/half-native-shared68/transport-synthetic-r3/scripts/nccl_root_model_move.py'
    cases={'z1':('unitree_z1_stackbox',12),'dual':('unitree_z1_dual_arm_stackbox',7),'pencils':('unitree_z1_dual_arm_cleanup_pencils',8)}
    pairs={}
    for alias,(case,rounds)in cases.items():
        original=repo/('experiments/nonv2-all-blocks59/full63-r1'if alias=='z1'else f'experiments/nonv2-other-blocks63/{alias}-full-r1')
        original_manifest=load(original/'production-sources.json')
        rows={};manifests={};guards={}
        for variant,count in(('baseline',63),('candidate',64)):
            label=f'{alias}_{variant}_full_r1';folder=code/label;overlay=folder/'production-overlay';out=evidence/label
            manifest=load(overlay/'nccl-parallel-production-sources.json');assert len(manifest)==count
            for name,digest in manifest.items():
                p=overlay/name;assert sha(p if p.exists()else base/name)==digest,name
            for name,digest in load(folder/'asset-hashes.json').items():
                p=overlay/name;assert sha(p if p.exists()else base/name)==digest,name
            guard=load(folder/'guard-source-hashes.json')
            for name,digest in guard.items():assert sha(folder/'guard'/name)==digest,name
            guards[variant]=guard;manifests[variant]=manifest
            flags=load(folder/'guard/flags.json')
            assert flags==dict(load(base.parent/(case+'-flags.json')),UNIFOLM_NCCL_PARALLEL='nonv2-all-blocks')
            summary=load(out/'summary.json');assert all(summary[k]for k in('clean','quality_passed','workload_complete','formal_videos_exact','rank_videos_exact','original_process_timer','all_steps_and_rounds_preserved','no_inference_math_changes'))
            assert not summary['official_adopted']and len(summary['gpu_candidates'])==2
            row=next(r for r in load(out/'result.json')['cases']if r['case']==case)
            assert row['rounds']==rounds and row['videos']==2*rounds+1
            assert row['process_wall_s']==summary['process_wall_s']and row['psnr_db']==summary['psnr_db']
            assert row['execution_ok']and row['profile_complete']and row['psnr_passed']
            parallel=load(out/'parallel/parallel-complete.json')
            assert parallel['stage_calls']==dict(decision=50*rounds,world=50*rounds)
            assert parallel['GPUs']==2 and parallel['rank_videos_SHA256_equal']and len(parallel['video_hashes'])==2*rounds+1
            comparison=load(out/'original-formal-video-comparison.json')
            assert comparison['all_original_formal_SHA_equal']and comparison['current_hashes']==comparison['original_formal_hashes']==parallel['video_hashes']
            for rank in(0,1):
                report=load(out/f'parallel/rank{rank}-controller.json')
                assert report['stage_captures']==dict(decision=33,world=33)and report['controller']['captured']==66
                if variant=='candidate':
                    transport=load(out/f'parallel/rank{rank}-weight-transport.json')
                    assert all(transport[k]for k in('root_only_H2D','no_precision_change','original_parameter_objects_and_aliases','original_contiguous_strides','all_tensor_starts_256_byte_aligned','standard_cuda_lifecycle_after_broadcast','no_operand_export'))
                    assert not transport['fallback_native_cuda']
            profile=load(out/f'profiles/{case}.json')
            assert profile['total_process_wall_s']==row['process_wall_s']
            rows[variant]=dict(process_wall_s=row['process_wall_s'],model_to_gpu_s=profile['stages']['model_to_gpu']['wall_s'],psnr_db=row['psnr_db'],GPUs=2)
        assert manifests['baseline']==original_manifest
        assert {n for n,h in original_manifest.items()if manifests['candidate'][n]!=h}=={runtime}
        assert set(manifests['candidate'])-set(original_manifest)=={mover}
        assert manifests['candidate'][mover]==sha(qualified)
        assert guards['baseline']==guards['candidate']
        assert rows['baseline']['psnr_db']==rows['candidate']['psnr_db']
        pairs[alias]=dict(**rows,saved_s=round(rows['baseline']['process_wall_s']-rows['candidate']['process_wall_s'],6))
    public=load(evidence/'public-repro-cpu-r1/summary.json')
    assert public['passed']and public['no_GPU_execution']and len(public['cases'])==3
    assert public['public_script_SHA256']==sha(repo/'reproduce/run_nonv2_root64.py')
    assert all(r['source_count']==64 and r['exact_timed_manifest_and_assets']for r in public['cases'])
    assert pairs==load(evidence/'paired-process-comparison.json')['cases']
    print('Root64: three complete two-GPU AB pairs, unchanged inference sources, video hashes and public reconstruction verified.')


if __name__=='__main__':verify()
