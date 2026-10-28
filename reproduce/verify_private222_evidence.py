"""Check frozen222, matched full AB, storage qualification and official103."""
import ast
import hashlib
import json
from pathlib import Path
import re


def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    root=Path(__file__).resolve().parents[1];storage=root/'experiments/private-nccl-tmpfs';comm=root/'results/aggressive-20261028/private-nccl-tmpfs'
    for label in('cpu-r1','graph-r1'):
        code=storage/label
        for n,h in load(code/'source-hashes.json').items():assert sha(code/n)==h
    cpu=load(comm/'cpu-r1/cpu-report.json');assert cpu['passed']and cpu['no_GPU_execution']and cpu['unrelated_paths_unchanged']and cpu['disabled_transparent']and cpu['nonprivate_root_rejected']and cpu['own_files_cleaned']
    assert len(cpu['records'])==2 and all(r['child_mmap_exact']and r['unlink_exact']for r in cpu['records'])
    adapter=load(storage/'qualified-adapter.json');assert sha(storage/'cpu-r1/nccl_private_tmpfs.c')==adapter['source_sha256']
    public=load(comm/'public-repro-cpu-r1/summary.json');assert public['passed']and public['no_GPU_execution']and public['matches_qualified_native_binary']
    for n,h in public['public_source_hashes'].items():assert sha(storage/n)==h
    assert load(comm/'public-repro-cpu-r1/cpu-report.json')['passed']
    graph=load(comm/'graph-r1/nccl-geglu-model-report.json');assert graph['clean']and len(graph['ranks'])==2
    for rank in graph['ranks']:
        assert rank['all_gather_exact']and rank['graph_exact']
        assert [r['local_MiB']for r in rank['records']]==[1,10,25,40]
        assert all(r['exact_copy']for r in rank['records'])
    family=root/'experiments/initial-boundary-221/private-tmpfs-full';ev=root/'results/aggressive-20261028/initial-boundary-221/private-tmpfs-full'
    pair=load(ev/'complete-process-AB.json');assert pair['same_GPU_pair']and pair['same_guard_scripts']and pair['same_preload_binary']and pair['same_private_tmpfs_storage']and pair['original_evaluator_timer_preserved']and pair['one_full_AB_not_six_full_process_pairs']
    case='unitree_z1_dual_arm_stackbox_v2';flags=[];production=[]
    for variant,count in [('baseline',221),('candidate',222)]:
        code=family/f'full-{variant}-r1';out=ev/f'full-{variant}-r1'
        for n,h in load(code/'source-hashes.json').items():assert sha(code/n)==h
        manifest=load(code/'production-sources.json');assert len(manifest)==count;production.append(manifest)
        frozen=root/('experiments/initial-boundary-221/production222'if variant=='candidate'else'experiments/temporal-block-220/production221')
        assert manifest==load(frozen/'production-sources.json')
        for n,h in load(root/'unifolm/world-head-filter-cache-production-sources.json').items():assert manifest[n]==h
        summary=load(out/'summary.json');assert all(summary[k]for k in('clean','workload_complete','quality_passed','baseline_videos_exact','rank_videos_exact','original_evaluator_timer_unchanged'))
        assert summary['gpu_candidates']==['0','1']
        row=next(c for c in load(out/'result.json')['cases']if c['case']==case)
        assert (row['rounds'],row['videos'])==(11,23)and row['execution_ok']and row['profile_complete']and row['psnr_passed']
        assert row['process_wall_s']==summary['process_wall_s']and row['psnr_db']==29.675324786148643
        assert load(out/f'profiles/{case}.json')['total_process_wall_s']==row['process_wall_s']
        assert re.findall(r'^TIMING case=(\S+) process_wall_s=([0-9.]+)$',(out/'runtime.log').read_text(),re.M)==[(case,str(row['process_wall_s']))]
        proof=load(out/'parallel/parallel-complete.json');assert proof['rank_videos_SHA256_equal']and proof['video_count']==23 and proof['stage_calls']==dict(decision=550,world=550)
        comparison=load(out/'baseline-video-comparison.json');assert comparison['all23_SHA256_equal']and comparison['current_hashes']==comparison['baseline_hashes']and len(comparison['current_hashes'])==23
        for rank in(0,1):
            controller=load(out/f'parallel/rank{rank}-controller.json');assert controller['stage_captures']==dict(decision=16,world=32)
            if variant=='candidate':
                initial=controller['controller']['initial_block'];assert initial['captured']==1 and initial['rows'][0]['grouped_copy_nodes']==2 and initial['rows'][0]['additional_collectives']==0
        backing=load(out/'storage-proof.json');assert backing['passed']and backing['all_live_NCCL_backing_on_private_tmpfs']and backing['dev_shm_device']!=backing['private_tmpfs_device']
        assert all(r['backing_files']and all(f['device']==backing['private_tmpfs_device']for f in r['backing_files'])for r in backing['rank_control_files'])
        flags.append(load(out/'flags.json'))
    for n in('wait_nccl_full209.py','retry_nccl_full221.py','check_dual_gpu_graph.py'):assert sha(family/'full-baseline-r1'/n)==sha(family/'full-candidate-r1'/n)
    assets=[load(family/('full-'+v+'-r1')/'timing-asset-hashes.json')for v in('baseline','candidate')]
    for n in('scripts/evaluation/world_model_interaction.py','scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):assert assets[0][n]==assets[1][n]
    assert {k:v for k,v in flags[0].items()if k!='UNIFOLM_NCCL_PARALLEL'}=={k:v for k,v in flags[1].items()if k!='UNIFOLM_NCCL_PARALLEL'}
    assert pair['complete']['baseline']['process_wall_s']==272.043547 and pair['complete']['candidate']['process_wall_s']==268.844229
    assert abs(pair['same_GPU_full_AB_saved_s']-3.199318)<1e-6
    adopted=load(root/'experiments/initial-boundary-221/production222/accepted-outcome.json');assert adopted['official_adopted']and adopted['all103_officially_validated']and adopted['all23_videos_SHA_equal']
    assert adopted['source_count']==222 and adopted['V2_s']==268.844229 and adopted['five_case_sum_s']==816.711804
    historical=root/'results/history/official222-z1-fused-before-blocks63'
    official=ev/'official103';final=load(official/'result.json');assert final['status']=='accepted'and final['psnr_validation']['video_count']==103 and final==load(historical/'result.json')
    for n,h in adopted['official_pair_sha256'].items():assert sha(historical/n)==sha(official/n)==h
    current=next(r for r in load(root/'results/history/official-other63-before-decision223/result.json')['cases']if r['case']==case)
    assert current['process_wall_s']==adopted['V2_s']and current['psnr_db']==29.675324786148643
    copies=load(official/'immutable-video-copies.json');assert copies['count']==103 and copies['ordinary_files']and copies['original_outputs_unchanged']and copies['SHA_verified_same_FS_links_or_cross_FS_copies']
    for n,h in load(official/'script-hashes.json').items():assert sha(official/n)==h
    parser=ast.parse((root/'reproduce/run_v2_parallel.py').read_text());assert any(isinstance(n,ast.Constant)and n.value==flags[1]['UNIFOLM_NCCL_PARALLEL']for n in ast.walk(parser))
    print('Adopted222: matched full AB, original209 arithmetic, private tmpfs CPU/graph and official103 verified.')


if __name__=='__main__':verify()
