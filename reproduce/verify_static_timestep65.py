"""Verify static-only cache fidelity evidence and immutable failed GPU attempts."""
import hashlib,json
from pathlib import Path


def verify():
 repo=Path(__file__).resolve().parents[1];root=repo/'experiments/nonv2-static-timestep65';evidence=repo/'results/aggressive-20261029/nonv2-static-timestep65'
 load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 baseline=repo/'experiments/nonv2-vae-all-display/unifolm';original=load(repo/'experiments/nonv2-all-blocks59/full63-r1/production-sources.json')
 primitive=repo/'experiments/static-timestep-cache/cpu-r1'
 for n,h in load(primitive/'source-hashes.json').items():assert sha(primitive/n)==h
 assert '2400_full_outputs_bitwise=true'in(primitive/'cpu.log').read_text()
 summary=load(repo/'results/aggressive-20261029/static-timestep-cache/cpu-r1/summary.json');assert summary['passed']and summary['no_GPU_execution']and summary['full_CPU_outputs_bitwise']==2400 and summary['stale_cache_rejections']==7
 for revision in(1,2,3,4):
  frozen=root/f'full-r{revision}';overlay=frozen/'production-overlay';manifest=load(frozen/'production-sources.json')
  assert len(manifest)==65 and {n for n,h in original.items()if manifest[n]!=h}=={'src/unifolm_wma/utils/cuda_graph.py','src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py'}
  assert set(manifest)-set(original)=={'src/unifolm_wma/utils/static_timestep_cache.py','src/unifolm_wma/utils/static_timestep_runtime.py'}
  for n,h in manifest.items():
   p=overlay/n;assert sha(p if p.exists()else baseline/n)==h,n
  for n,h in load(frozen/'timing-asset-hashes.json').items():assert sha(overlay/n)==h,n
  for n,h in load(frozen/'guard-source-hashes.json').items():assert sha(frozen/n)==h,n
  assert load(frozen/'flags.json')==load(repo/'experiments/nonv2-all-blocks59/full63-r1/flags.json')
  assert load(frozen/'trial.json')['per_role_expected_gathers']==dict(decision_fill=33,decision_read=33,world_fill=33,world_read=33)
  runtime=(overlay/'src/unifolm_wma/utils/static_timestep_runtime.py').read_text()
  assert "if type(module)is ResBlock"in runtime and "'time_embed':backbone.time_embed"in runtime and "modules['fps_embedding']=backbone.fps_embedding"in runtime
  assert 'self.executor(*args,**kwargs)'in runtime and 'timesteps=args[3]'in runtime
  assert 'all200_outputs_bitwise=true all200_parallel_executor_calls=true dynamic_convolution_live=true no_GPU=true'in(frozen/'crossstep-cpu.log').read_text()
  cpu=load(evidence/f'public-repro-cpu-r{revision}/summary.json');assert cpu['passed']and cpu['no_GPU_execution']and cpu['source_count']==65 and cpu['exact_timed_manifest_and_assets']and cpu['actual_WMA_ResBlock']
  runner='run_nonv2_static_timestep65.py'if revision==1 else f'run_nonv2_static_timestep65_r{revision}.py'
  assert cpu['public_script_SHA256']==sha(repo/'reproduce'/runner)
  if revision==1:
   failure=load(evidence/'full-r1/summary.json');assert not failure['clean']and failure['process_wall_s']is None and failure['GPU_capture_index_copy_failure']
   log=(evidence/'full-r1/attempt-2/benchmark.log').read_text();assert 'index_copy_'in log and 'operation not permitted when stream is capturing'in log
  if revision==2:
   failure=load(evidence/'full-r2/summary.json');assert not failure['clean']and failure['process_wall_s']is None
   assert 'No space left on device'in(evidence/'full-r2/launch.log').read_text()
  if revision in(3,4):
   parallel=(overlay/'src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py').read_text()
   assert "os.environ['TEAM07_STATIC_TIME_CACHE']"in parallel
   if revision==3:assert "TRITON_CACHE_MANAGER='split_experiment_triton_cache:ReadWarmSplitCache'"in parallel
   else:
    assert "TRITON_CACHE_MANAGER="not in parallel
    cache=(overlay/'src/unifolm_wma/utils/static_timestep_cache.py').read_text()
    assert 'with kernel_cache_scope():_store_slot'in cache and 'with kernel_cache_scope():_load_slot'in cache
    assert "cache.__dict__['__cache_cls']=previous_class"in cache
    assert 'STATIC_CACHE_SCOPE_CPU passed=true environment_and_Triton_class_restored=true original_FFN_assertion_retained=true'in(frozen/'crossstep-cpu.log').read_text()
   manager=overlay/'scripts/evaluation/split_experiment_triton_cache.py'
   assert sha(manager)==sha(repo/'experiments/temporal-live-trials/z1-both-r2/guard/split_experiment_triton_cache.py')
  if revision==3 and(evidence/'full-r3/summary.json').exists():
   failure=load(evidence/'full-r3/summary.json');assert not failure['clean']and failure['process_wall_s']is None
   log=(evidence/'full-r3/launch.log').read_text();assert "assert 'TRITON_CACHE_MANAGER' not in os.environ"in log
 full=evidence/'full-r4';result=load(full/'summary.json')
 assert result['clean']and result['quality_passed']and result['workload_complete']and result['formal_videos_exact']and not result['official_adopted']
 assert result['process_wall_s']==157.446954 and result['psnr_db']==46.553398187472595
 comparison=load(full/'original-formal-video-comparison.json');assert comparison['all25_original_formal_SHA_equal']and len(comparison['current_hashes'])==25 and comparison['current_hashes']==comparison['original_formal_hashes']
 for rank in(0,1):
  report=load(full/f'parallel/rank{rank}-controller.json');feature=report['feature_reuse']
  assert report['stage_captures']==dict(decision=66,world=66)and report['controller']['captured']==132
  assert {role:len(names)for role,names in report['graph_role_modules'].items()}==dict(decision_fill=33,decision_read=33,world_fill=33,world_read=33)
  assert feature['graph_count']==4 and len(feature['completed_generations'])==24
  assert feature['counts']==dict(decision_calls=600,decision_fill=50,world_calls=600,world_fill=50,decision_read=550,world_read=550)
  assert not feature['precision_changed']and not feature['dynamic_features_cached']and not feature['sampler_steps_reduced']and not feature['rounds_reduced']
  assert feature['graph_copy_selfcheck']['passed']and feature['graph_copy_selfcheck']['bitwise_graph_copy_slots']==100
  assert len(feature['modules'])==24 and len(feature['capture_rows'])==96
 print('Static timestep65: CPU fidelity/four public rebuilds and failed attempts retained;full12/25 exact videos,100 graph-copy slots and132 capture records verified;157.446954s is slower and unadopted.')

if __name__=='__main__':verify()
