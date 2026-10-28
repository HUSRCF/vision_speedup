import json,os
from pathlib import Path
import torch
import torch.utils.checkpoint
from unifolm_wma.utils import world_persistent_zero_low_runtime as r
assert not torch.cuda.is_initialized()
dataset='unitree_z1_dual_arm_stackbox_v2';env={r.FLAG:'1',**r.DEPENDENCIES}
assert r.configured(env,'native',dataset)
assert not r.configured({},'native',dataset) and not r.configured({r.FLAG:'1'},'native','other')
for raw in ('bad','true','2'):
 try:r.configured({r.FLAG:raw},'native',dataset)
 except ValueError:pass
 else:raise AssertionError('flag')
for name in r.DEPENDENCIES:
 bad=dict(env);bad.pop(name)
 try:r.configured(bad,'native',dataset)
 except ValueError:pass
 else:raise AssertionError(name)
try:r.configured(env,'half',dataset)
except ValueError:pass
else:raise AssertionError('mode')
assert sum(v['nodes_by_stage']['world'] for v in r.CONFIGS)==77
assert all(set(v['nodes_by_stage'])=={'world'} for v in r.CONFIGS)
calls=[];sentinel=object()
class FakeJIT:
 arg_names=['XH','XL','WH','WL','LowCount','Bias','Y','M','N','K','HAS_BIAS','BM','GROUP','OUTPUT_GROUPS']
 def run(self,*args,**kwargs):calls.append((args,kwargs));return sentinel
jit=FakeJIT();parent=r.parent
saved=(parent.ORIGINAL_JIT,parent.ORIGINAL_RUN,torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
try:
 parent.ORIGINAL_JIT=jit;parent.ORIGINAL_RUN=jit.run
 os.environ.update(env);model=torch.nn.Module().eval()
 controller=r.configure_world_persistent_zero_low(model,mode='native',dataset=dataset)
 assert r.enabled and controller is r.active and jit.run==r.original_outer
 try:r.configure_world_persistent_zero_low(model,mode='native',dataset=dataset)
 except RuntimeError:pass
 else:raise AssertionError('duplicate configure')
 try:r.install(model,r.CONFIGS)
 except AssertionError:pass
 else:raise AssertionError('duplicate install')
 with torch.no_grad():
  torch.backends.cuda.matmul.allow_tf32=False
  for row in r.CONFIGS:
   args=tuple(torch.empty(0) for _ in range(7))+tuple(row['geometry'])+(row['has_bias'],row['baseline_config'][0],row['baseline_config'][2],row['output_groups'])
   meta=dict(num_warps=row['baseline_config'][3],num_stages=row['baseline_config'][4],enable_fp_fusion=False)
   for decision,warmup,on in ((True,False,True),(False,False,True),(False,False,False),(False,True,True)):
    torch.backends.cudnn.allow_tf32=decision
    with r.mode(on):assert r.launched(*args,grid=(123,),warmup=warmup,**meta) is sentinel
    actual,kw=calls[-1];assert all(a is b for a,b in zip(actual,args)) and kw==dict(meta,grid=(123,),warmup=warmup)
  try:
   with r.mode(False):raise RuntimeError('context restoration')
  except RuntimeError:pass
  assert r.enabled and controller.report()['captured']==0 and parent.ORIGINAL_RUN is r.launched
 r.uninstall();assert parent.ORIGINAL_RUN==jit.run and not r.enabled
 assert not torch.cuda.is_initialized()
finally:
 if r.active is not None:r.uninstall()
 parent.ORIGINAL_JIT,parent.ORIGINAL_RUN,torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32=saved
report=dict(passed=True,invalid_flags=3,dependency_guards=len(r.DEPENDENCIES),mode_guard=True,world_only_77_nodes=True,duplicate_rejected=True,original_JIT_and_outer_run_restored=True,decision_CPU_disabled_and_warmup_fallback_arguments_unchanged=True,exception_context_restored=True,CUDA_context_initialized=False)
Path('world-persistent-zero-low-production-cpu.json').write_text(json.dumps(report,indent=2)+chr(10))
print(json.dumps(report))
