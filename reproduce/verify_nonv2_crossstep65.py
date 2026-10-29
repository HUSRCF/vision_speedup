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
 print('NonV2 crossstep65: original63 math helpers unchanged; role-specific module sets,1200 delegated CPU calls and public reconstruction verified; no GPU quality/speed claim.')

if __name__=='__main__':verify()
