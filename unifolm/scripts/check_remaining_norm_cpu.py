import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from unifolm_wma.utils import world_two_product_tiles_runtime as r
from unifolm_wma.utils import triton_two_product_linear as original
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
os.environ.update(env);model=SimpleNamespace();saved=r.ORIGINAL_RUN;jit=original._two_product_gemm
assert jit is r.ORIGINAL_JIT and jit.run==saved
controller=r.configure_two_product_tiles(model,mode='native',dataset=dataset)
assert original._two_product_gemm is jit and jit.run==controller.run and controller.enabled
try:r.configure_two_product_tiles(model,mode='native',dataset=dataset)
except RuntimeError:pass
else:raise AssertionError('duplicate')
try:r.install(r.CONFIGS)
except AssertionError:pass
else:raise AssertionError('install duplicate')
# Spy only on the saved fallback target, never execute GPU JIT on CPU tensors.
calls=[]
def spy(*args,**kwargs):calls.append((args,kwargs));return 'forwarded'
r.ORIGINAL_RUN=spy
args=(torch.empty(1),)*5+(40960,320,320,True,True,64,64,64,32)
kw=dict(grid=(3200,),num_warps=4,num_stages=2,enable_fp_fusion=False,warmup=False)
with torch.no_grad():
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 assert jit.run(*args,**kw)=='forwarded' and controller.fallbacks['storage']==1
 assert all(a is b for a,b in zip(calls[-1][0],args)) and calls[-1][1]==kw
 with r.mode(False):assert jit.run(*args,**kw)=='forwarded'
 torch.backends.cudnn.allow_tf32=True;assert jit.run(*args,**kw)=='forwarded';torch.backends.cudnn.allow_tf32=False
 assert jit.run(*args,**dict(kw,num_stages=3))=='forwarded'
 bad=args[:5]+(19,23,13,True,True,64,64,64,32);assert jit.run(*bad,**kw)=='forwarded'
assert not controller.captures
r.ORIGINAL_RUN=saved;r.uninstall();assert original._two_product_gemm is jit and jit.run==saved
assert not torch.cuda.is_initialized()
report=dict(passed=True,invalid_flags=3,dependency_guards=len(r.DEPENDENCIES),mode_guard=True,duplicate_rejected=True,original_run_argument_forwarding=True,JIT_object_and_alias_preserved=True,disabled_stage_storage_launch_geometry_fallbacks=True,original_run_restored=True,CUDA_context_initialized=False)
Path('remaining-norm-production-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
