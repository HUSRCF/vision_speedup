"""Verify the prepared dual-GPU temporal integration without inferring GPU gains."""
import hashlib,json
from pathlib import Path


def verify():
 repo=Path(__file__).resolve().parents[1];f=repo/'experiments/nonv2-crossstep65/full-r1';overlay=f/'production-overlay';base=repo/'experiments/nonv2-vae-all-display/unifolm'
 load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 manifest=load(f/'production-sources.json');original=load(repo/'experiments/nonv2-all-blocks59/full63-r1/production-sources.json')
 runtime='src/unifolm_wma/utils/nccl_nonv2_all_blocks_runtime63.py';feature='src/unifolm_wma/utils/crossstep_feature_runtime.py';reuse='src/unifolm_wma/utils/crossstep_feature_reuse.py';graph='src/unifolm_wma/utils/cuda_graph.py'
 assert len(manifest)==65 and {n for n,h in original.items()if manifest[n]!=h}=={runtime,graph}
 assert set(manifest)-set(original)=={feature,reuse}
 for n,h in manifest.items():
  p=overlay/n;assert sha(p if p.exists()else base/n)==h,n
 for n,h in load(f/'timing-asset-hashes.json').items():assert sha(overlay/n)==h,n
 for n,h in load(f/'guard-source-hashes.json').items():assert sha(f/n)==h,n
 assert load(f/'flags.json')==load(repo/'experiments/nonv2-all-blocks59/full63-r1/flags.json')
 assert load(f/'trial.json')['per_role_expected_gathers']==dict(decision_full=33,decision_reuse=31,world_full=33,world_reuse=31)
 entry=(overlay/'scripts/evaluation/world_model_interaction.py').read_text()
 assert entry.index('PROCESS_START_TIME =')<entry.index('_nccl_early_start()')<entry.index('_feature_runtime.install(model,')
 assert entry.index('_feature_runtime.finish(args)')<entry.index('_nccl_finish(args)')<entry.index('timing.write(args.profile_path)')
 source=(overlay/runtime).read_text()
 assert "assert _graph_captures==expected"in source and "len(_denoiser_graph.entries)==4"in source
 assert 'with _mode(True):result=original(*args,**kwargs)'in source
 single=repo/'experiments/temporal-live-trials/z1-both-r2/production-overlay'
 assert sha(overlay/reuse)==sha(single/reuse)
 expected=(single/feature).read_text().replace('self.model=model;self.graph=model.apply_model;assert isinstance(self.graph,DenoiserGraph) and not self.graph.entries','self.model=model;self.executor=model.apply_model;self.graph=self.executor.denoiser_graph;assert isinstance(self.graph,DenoiserGraph) and not self.graph.entries').replace('result=self.graph(*args,**kwargs)','result=self.executor(*args,**kwargs)')
 assert (overlay/feature).read_text()==expected
 cpu=(f/'crossstep-cpu.log').read_text();assert 'passed=true all1200_calls_through_parallel_executor=true role_module_sets=true duplicate_rejected=true no_GPU=true'in cpu
 result=load(repo/'results/aggressive-20261029/nonv2-crossstep65/public-repro-cpu-r1/summary.json')
 assert result['passed']and result['no_GPU_execution']and result['source_count']==65 and result['exact_timed_manifest_and_assets']and result['all1200_calls_through_parallel_executor']
 assert result['public_script_SHA256']==sha(repo/'reproduce/run_nonv2_crossstep65.py')
 evidence=repo/'results/aggressive-20261029/nonv2-crossstep65'
 full=load(evidence/'full-r1/summary.json');assert full['clean']and full['workload_complete']and full['quality_passed']and full['rank_videos_exact']
 assert full['process_wall_s']==150.838389 and full['psnr_db']==26.081602114464168
 complete=load(evidence/'full-r1/parallel/parallel-complete.json');assert complete['stage_calls']==dict(decision=600,world=600)and complete['rank_videos_SHA256_equal']and len(complete['video_hashes'])==25
 for rank in(0,1):
  report=load(evidence/f'full-r1/parallel/rank{rank}-controller.json');feature_report=report['feature_reuse']
  assert report['stage_captures']==dict(decision=64,world=64)and report['controller']['captured']==128
  assert {role:len(names)for role,names in report['graph_role_modules'].items()}==dict(decision_full=33,decision_reuse=31,world_full=33,world_reuse=31)
  assert feature_report['graph_count']==4 and len(feature_report['completed_generations'])==24
  assert feature_report['counts']['decision_calls']==feature_report['counts']['world_calls']==600
  assert feature_report['counts']['decision_reuse']==feature_report['counts']['world_reuse']==180
  assert not feature_report['round_count_reduced']and not feature_report['sampler_step_count_reduced']
 accepted=load(repo/'experiments/nonv2-crossstep65/accepted-outcome.json');fresh=load(evidence/'official103/validated/result.json');formal=load(repo/'results/history/official-before-prediction65/result.json')
 assert accepted['official_adopted']and accepted['official103_accepted']and accepted['GPUs']==2
 assert accepted['case_sum_s']==764.366152 and accepted['formal_saved_s']==1.364890 and accepted['historical_comparison_not_matched_AB']
 assert fresh==formal and fresh['status']=='accepted'and fresh['psnr_validation']['video_count']==103
 history=repo/'results/history/official-before-crossstep65'
 for n,h in accepted['pair_hashes'].items():assert sha(repo/'results/history/official-before-prediction65'/n)==sha(evidence/'official103/validated'/n)==h
 for n,h in accepted['previous_pair_hashes'].items():assert sha(history/n)==h
 before=load(history/'result.json')
 for row in formal['cases']:
  old=next(c for c in before['cases']if c['case']==row['case'])
  if row['case']=='unitree_z1_stackbox':assert row['process_wall_s']==full['process_wall_s']and row['psnr_db']==full['psnr_db']
  else:assert row==old
 copies=load(evidence/'official103/validated/immutable-video-copies.json');prior=load(repo/'results/aggressive-20261029/half-native-shared68/official103/validated/immutable-video-copies.json')
 other=lambda data:{p:h for p,h in data['hashes'].items()if'/unitree_z1_stackbox/'not in p}
 assert copies['count']==103 and copies['ordinary_files']and copies['original_outputs_unchanged']
 assert len(other(copies))==78 and other(copies)==other(prior)
 print('Adopted nonV2 crossstep65: full12/25 clean two-rank reuse,128 role-specific captures,fresh official103 and unchanged other78 video hashes verified.')

if __name__=='__main__':verify()
