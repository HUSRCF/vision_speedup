import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from unifolm_wma.utils import triton_split_group_norm as original
from unifolm_wma.utils.serial_group_norm_runtime import FLAG,configured,configure_serial_norm,uninstall,ORIGINAL
assert not torch.cuda.is_initialized()
dataset='unitree_z1_dual_arm_stackbox_v2'
assert not configured({},'native',dataset)
assert configured({FLAG:'1','UNIFOLM_SPLIT_GROUP_NORM':'fp32'},'native',dataset)
assert not configured({FLAG:'1'},'native','other')
for raw in ('bad','true','2'):
 try:configured({FLAG:raw},'native',dataset)
 except ValueError:pass
 else:raise AssertionError('flag')
for mode,policy in [('half','fp32'),('native','fp16'),('native','0')]:
 try:configured({FLAG:'1','UNIFOLM_SPLIT_GROUP_NORM':policy},mode,dataset)
 except ValueError:pass
 else:raise AssertionError('policy')
norm=torch.nn.GroupNorm(32,64).eval();value=torch.randn(2,64,7,9)
keys=list(norm.state_dict());pointers=[p.data_ptr() for p in norm.parameters()];saved=[p.detach().clone() for p in norm.parameters()]
with torch.no_grad():expected=ORIGINAL(value,norm)
os.environ[FLAG]='1';os.environ['UNIFOLM_SPLIT_GROUP_NORM']='fp32'
model=SimpleNamespace();controller=configure_serial_norm(model,mode='native',dataset=dataset)
with torch.no_grad():actual=original.split_group_norm_silu(value,norm)
assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
assert controller.fallbacks=={'input':1} and not controller.captures
try:configure_serial_norm(model,mode='native',dataset=dataset)
except RuntimeError:pass
else:raise AssertionError('duplicate')
assert list(norm.state_dict())==keys and pointers==[p.data_ptr() for p in norm.parameters()]
assert all(torch.equal(p,s) for p,s in zip(norm.parameters(),saved))
uninstall();assert original.split_group_norm_silu is ORIGINAL and not torch.cuda.is_initialized()
report=dict(passed=True,invalid_flags=3,dependency_guards=3,CPU_original_fallback=True,duplicate_rejected=True,state_keys_pointers_values_unchanged=True,CUDA_context_initialized=False)
Path('serial-norm-production-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
