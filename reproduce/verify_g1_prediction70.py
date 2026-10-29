"""Verify G1 reuse source identity, unchanged native bindings and full failures."""
import ast
import hashlib
import json
from pathlib import Path


def verify():
    repo=Path(__file__).resolve().parents[1]
    load=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    base=repo/'experiments/nonv2-vae-all-display/unifolm'
    original_root=repo/'experiments/half-native-shared68/full-r2/production-overlay'
    original=load(original_root/'nccl-parallel-production-sources.json')
    runtime='src/unifolm_wma/utils/nccl_nonv2_native_shared_runtime68.py'
    graph='src/unifolm_wma/utils/cuda_graph.py'
    feature='src/unifolm_wma/utils/crossstep_feature_runtime.py'
    predictor='src/unifolm_wma/utils/crossstep_feature_reuse.py'
    entry='scripts/evaluation/world_model_interaction.py'
    old_feature=repo/'experiments/nonv2-late23-prediction65/full-r1/production-overlay'
    evidence=repo/'results/aggressive-20261029/g1-prediction70'
    cpu_root=evidence/'public-repro-cpu-r1';cpu=load(cpu_root/'summary.json')
    assert cpu['passed']and cpu['no_GPU_execution']
    assert cpu['public_script_SHA256']==sha(repo/'reproduce/run_g1_prediction70.py')
    for name,h in cpu['logs_SHA256'].items():assert sha(cpu_root/name)==h
    assert {row['depth']for row in cpu['trials']}=={13,23}
    for depth,seconds,psnr,gpus in((13,153.741615,17.36695080270202,['2','3']),
                                  (23,138.975980,18.4700791121282,['0','1'])):
        frozen=repo/f'experiments/g1-prediction70/deep{depth}-r1';overlay=frozen/'production-overlay'
        manifest=load(overlay/'nccl-parallel-production-sources.json')
        assert len(manifest)==70 and set(manifest)-set(original)=={feature,predictor}
        assert {n for n,h in original.items()if manifest[n]!=h}=={runtime,graph}
        for name,h in manifest.items():
            path=overlay/name
            assert sha(path if path.exists()else base/name)==h,name
        for name,h in load(frozen/'asset-hashes.json').items():
            path=repo/'unifolm'/name if name.endswith('.so')else overlay/name
            assert sha(path)==h,name
        for name,h in load(frozen/'guard-source-hashes.json').items():assert sha(frozen/'guard'/name)==h
        for name in(feature,predictor):assert sha(overlay/name)==sha(old_feature/name)
        assert (overlay/graph).read_text()==(base/graph).read_text().replace(
            'with torch.cuda.graph(graph, stream=stream):',
            'with torch.cuda.graph(graph, stream=stream, pool=getattr(self, "experimental_capture_pool", None)):')
        metadata=load(frozen/'trial.json');policy=metadata['feature_policy']
        assert policy==dict(input_start=6 if depth==13 else 1,output_end=6 if depth==13 else 11,
                            start=20,stop=40,boundary_residual=False,scope='deep',stages=['decision','world'])
        text=(overlay/entry).read_text()
        install=next(line for line in text.splitlines()if '_feature_runtime.install(model,'in line)
        literal=ast.literal_eval(install.split('**',1)[1][:-1]);literal['stages']=list(literal['stages'])
        assert literal==policy
        stripped=text.replace(install+'\n','').replace('    _feature_runtime.finish(args)\n','')
        stripped=stripped.replace('        from unifolm_wma.utils import crossstep_feature_runtime as _feature_runtime\n','')
        stripped=stripped.replace('    from unifolm_wma.utils import crossstep_feature_runtime as _feature_runtime\n','')
        assert stripped==(original_root/entry).read_text()
        assert not metadata['official_adopted']and metadata['original68_native_math_unchanged']
        proof=load(overlay/'nccl-entry-timing-proof.json')
        assert proof['all_rounds_and50_DDIM_updates_preserved']and not proof['new_native_argument_or_address_inspection']
        assert proof['full_graph_original31_native_FFN_bindings']and proof['reuse_graph_partition_disabled']
        row=next(row for row in cpu['trials']if row['depth']==depth)
        for name in('exact_timed_manifest_and_assets','all1200_calls_through_executor',
                     'predictor_affine3600_CPU_passed','lexical_cache_scope_passed',
                     'refresh960_reuse240_partition_modes_checked'):assert row[name]
        run=evidence/f'deep{depth}-r1';summary=load(run/'summary.json')
        assert summary['clean']and summary['workload_complete']and summary['rank_videos_exact']
        assert summary['process_wall_s']==seconds and summary['psnr_db']==psnr
        assert not summary['quality_passed']and not summary['official_adopted']
        assert summary['verified_source_count']==70 and summary['gpu_candidates']==gpus
        row=next(x for x in load(run/'result.json')['cases']if x['case']=='unitree_g1_pack_camera')
        assert (row['rounds'],row['videos'])==(11,23)
        assert (row['process_wall_s'],row['psnr_db'])==(seconds,psnr)and not row['psnr_passed']
        assert row['execution_ok']and row['profile_complete']
        complete=load(run/'parallel/parallel-complete.json')
        assert complete['GPUs']==2 and complete['rank_videos_SHA256_equal']and len(complete['video_hashes'])==23
        assert complete['stage_calls']==dict(decision=550,world=550)
        for rank in(0,1):
            report=load(run/f'parallel/rank{rank}-controller.json');controller=report['controller']
            assert report['stage_captures']==dict(decision=31,world=31)
            assert controller['captured']==controller['GEGLU']['captured']==62
            assert controller['native_linear_capture_count']==124 and len(controller['native_binding'])==2
            assert all(row['selected_FFNs']==31 and row['selected_native_GEMMs']==62 for row in controller['native_binding'])
            roles=report['graph_roles']
            assert {role:row['native_FFN_partitions']for role,row in roles.items()}==dict(decision_full=31,decision_reuse=0,world_full=31,world_reuse=0)
            assert all(row['partition_enabled']==role.endswith('_full')and row['native_binding_required']==role.endswith('_full')for role,row in roles.items())
            assert controller['spatial_shared_boundary']['captured']==30 and controller['spatial_shared_boundary']['attention_calls']==150
            assert controller['temporal_shared_boundary']['captured']==32 and controller['temporal_shared_boundary']['attention_calls']==64
            reuse=report['feature_reuse']
            assert reuse['graph_count']==4 and reuse['window']==[20,40]and len(reuse['selected_blocks'])==depth
            assert reuse['counts']==dict(decision_calls=550,decision_refresh=440,decision_reuse=110,world_calls=550,world_refresh=440,world_reuse=110)
            assert reuse['alpha_histogram']=={'1.0':22,'0.5':198}and reuse['graph_prediction_selfcheck']['passed']
            assert len(reuse['completed_generations'])==22 and all(x['steps']==50 for x in reuse['completed_generations'])
            assert not reuse['sampler_step_count_reduced']and not reuse['round_count_reduced']
            move=load(run/f'parallel/rank{rank}-weight-transport.json')
            assert move['root_only_H2D']and move['standard_cuda_lifecycle_after_broadcast']and move['all_tensor_starts_256_byte_aligned']
            assert move['tensors']==3576 and move['buckets']==67 and move['total_bytes']==16718118720
    # An independent all-refresh control isolates the new executor from prediction.
    frozen=repo/'experiments/g1-prediction70/refresh-control-r1';overlay=frozen/'production-overlay'
    manifest=load(overlay/'nccl-parallel-production-sources.json')
    assert len(manifest)==70 and {n for n in original if manifest[n]!=original[n]}=={runtime,graph}
    for name,h in manifest.items():
        path=overlay/name;assert sha(path if path.exists()else base/name)==h,name
    for name,h in load(frozen/'asset-hashes.json').items():
        path=repo/'unifolm'/name if name.endswith('.so')else overlay/name
        assert sha(path)==h,name
    for name,h in load(frozen/'guard-source-hashes.json').items():assert sha(frozen/'guard'/name)==h
    previous=repo/'experiments/g1-prediction70/deep13-r1/production-overlay'
    assert sha(overlay/predictor)==sha(previous/predictor)
    expected=(previous/feature).read_text().replace("report['counts']['world_reuse']>0","report['counts'].get('world_reuse',0)==0 and report['stages']==[]")
    assert (overlay/feature).read_text()==expected
    expected=(previous/runtime).read_text().replace('len(_denoiser_graph.entries)==4','len(_denoiser_graph.entries)==2')
    expected=expected.replace("feature['graph_count']==4 and feature['stages']==['decision','world']","feature['graph_count']==2 and feature['stages']==[]")
    expected=expected.replace("feature['counts']['decision_reuse']==feature['counts']['world_reuse']==10*args.n_iter","feature['counts'].get('decision_reuse',0)==feature['counts'].get('world_reuse',0)==0")
    expected=expected.replace('for reuse in(False,True)}','for reuse in(False,)}')
    assert (overlay/runtime).read_text()==expected
    assert (overlay/entry).read_text()==(previous/entry).read_text().replace("'stages': ('decision', 'world')","'stages': ()")
    assert load(frozen/'trial.json')['feature_policy']['stages']==[]
    cpu_root=evidence/'public-control-repro-cpu-r1';cpu=load(cpu_root/'summary.json')
    assert cpu['passed']and cpu['no_GPU_execution']
    assert cpu['public_script_SHA256']==sha(repo/'reproduce/run_g1_refresh_control70.py')
    assert len(cpu['trials'])==1 and cpu['trials'][0]['control_full1200_calls_without_predictions']
    assert cpu['trials'][0]['exact_timed_manifest_and_assets']
    for name,h in cpu['logs_SHA256'].items():assert sha(cpu_root/name)==h
    run=evidence/'refresh-control-r1';summary=load(run/'summary.json')
    assert summary['clean']and summary['quality_passed']and summary['workload_complete']and summary['rank_videos_exact']
    assert summary['full_refresh_fidelity_control']and summary['original68_23_videos_SHA256_equal']and not summary['official_adopted']
    assert summary['process_wall_s']==157.165205 and summary['psnr_db']==26.10715633179594
    complete=load(run/'parallel/parallel-complete.json')
    old=load(repo/'results/aggressive-20261029/half-native-shared68/full-r2/parallel/parallel-complete.json')
    assert len(complete['video_hashes'])==23 and complete['video_hashes']==old['video_hashes']
    assert complete['stage_calls']==dict(decision=550,world=550)and complete['rank_videos_SHA256_equal']
    for rank in(0,1):
        report=load(run/f'parallel/rank{rank}-controller.json');feature_report=report['feature_reuse']
        assert report['stage_captures']==dict(decision=31,world=31)and len(report['controller']['native_binding'])==2
        assert set(report['graph_roles'])=={'decision_full','world_full'}
        assert feature_report['stages']==[]and feature_report['graph_count']==2 and feature_report['persistent_feature_bytes']==0
        assert feature_report['counts']==dict(decision_calls=550,decision_refresh=550,world_calls=550,world_refresh=550)
        assert feature_report['alpha_histogram']=={}and len(feature_report['completed_generations'])==22
    for revision in(1,2):
        failure=load(evidence/f'preparation-failure-r{revision}/summary.json')
        assert failure['no_GPU_execution']and failure['corrected_in_final_preflight']
    formal=load(repo/'results/history/official-before-resblock-tail66/result.json')
    assert formal['status']=='accepted'and formal['psnr_validation']['video_count']==103
    assert next(row for row in formal['cases']if row['case']=='unitree_g1_pack_camera')['process_wall_s']==155.314304
    adopted=load(repo/'experiments/nonv2-late23-prediction65/dual-accepted-outcome.json')
    for name,h in adopted['pair_hashes'].items():assert sha(repo/'results/history/official-before-resblock-tail66'/name)==h
    print('G1_PREDICTION70_VERIFIED exact68_math_helpers=true full31_bindings_unchanged=true full23_videos_two_ranks=true both_quality_failed=true all_refresh_control_23_SHA_exact=true formal_pair_unchanged=true')


if __name__=='__main__':verify()
