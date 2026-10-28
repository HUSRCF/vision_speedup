import json,os
from pathlib import Path
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.attention import SpatialTransformer
from unifolm_wma.utils import out_projection_residual_runtime as r
from unifolm_wma.utils.triton_two_product_linear import _two_product_gemm
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
modules=[]
for _ in range(20):
 module=SpatialTransformer.__new__(SpatialTransformer);torch.nn.Module.__init__(module)
 module.use_linear=True;module.in_channels=320;module._channels_last_output=False
 module.norm=torch.nn.GroupNorm(32,320,eps=1e-6);module.proj_in=torch.nn.Linear(320,320);module.proj_out=torch.nn.Linear(320,320);module.transformer_blocks=torch.nn.ModuleList([]);modules.append(module)
model=torch.nn.ModuleList(modules).eval();saved=[(m.forward,m.proj_out.forward) for m in model];parent=_two_product_gemm.run
os.environ.update(env);controller=r.configure_out_projection_residual(model,mode='native',dataset=dataset)
assert controller.parent_run==parent and _two_product_gemm.run==parent and controller.enabled
try:r.configure_out_projection_residual(model,mode='native',dataset=dataset)
except RuntimeError:pass
else:raise AssertionError('duplicate configure')
try:r.install(model,r.CONFIGS)
except AssertionError:pass
else:raise AssertionError('duplicate install')
with torch.no_grad():
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 x=torch.arange(2560,dtype=torch.float32).view(2,320,2,2)*.001
 assert torch.equal(model[0](x),saved[0][0](x)) and controller.fallbacks['input']==1
 with r.mode(False):assert torch.equal(model[0](x),saved[0][0](x))
 torch.backends.cudnn.allow_tf32=True;assert torch.equal(model[0](x),saved[0][0](x));torch.backends.cudnn.allow_tf32=False
 y=torch.zeros(2,320);assert torch.equal(model[0].proj_out(y),saved[0][1](y))
 controller.current={'module':model[0],'residual':x,'fused':False};ones=torch.ones_like(x)
 assert torch.equal(controller.finish(model[0],ones,x),ones+x)
 controller.current.update(fused=True,output_ptr=ones.data_ptr());assert controller.finish(model[0],ones,x) is ones
 calls=[]
 def spy(*args,**kwargs):calls.append((args,kwargs));return 'forwarded'
 controller.parent_run=spy;controller.projection_call=True
 args=('sentinel',);kw={'token':object()};assert controller.run(*args,**kw)=='forwarded'
 assert calls[-1][0]==args and calls[-1][1]==kw
 controller.current=None;controller.projection_call=False;controller.parent_run=parent
assert controller.report()['captured']==0
r.uninstall();assert _two_product_gemm.run==parent
assert all(m.forward==f and m.proj_out.forward==p for m,(f,p) in zip(model,saved))
assert not torch.cuda.is_initialized()
report=dict(passed=True,invalid_flags=3,dependency_guards=len(r.DEPENDENCIES),mode_guard=True,duplicate_rejected=True,original_run_argument_forwarding=True,CPU_disabled_stage_fallbacks=True,all_module_forwards_restored=True,parent_JIT_run_restored=True,patched_only_final_return=True,fused_finish_skips_add_once_and_fallback_adds_once=True,CUDA_context_initialized=False)
Path('out-projection-residual-production-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
