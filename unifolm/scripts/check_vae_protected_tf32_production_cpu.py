import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from unifolm_wma.utils.vae_protected_tf32_runtime import FLAG,DEPS,configured_chunk,decode_for_stage
base={**DEPS,FLAG:'14'}
assert not torch.cuda.is_initialized()
assert configured_chunk({},'unitree_z1_dual_arm_stackbox_v2','simulation')==0
assert configured_chunk(base,'unitree_z1_dual_arm_stackbox_v2','simulation')==14
for chunk in ['1','4','8']:
 assert configured_chunk({**base,FLAG:chunk},'unitree_z1_dual_arm_stackbox_v2','simulation')==int(chunk)
for raw in ['bad','2','-1']:
 try:configured_chunk({**base,FLAG:raw},'unitree_z1_dual_arm_stackbox_v2','simulation')
 except ValueError:pass
 else:raise AssertionError(raw)
for k in DEPS:
 wrong={**base,k:'bad'}
 try:configured_chunk(wrong,'unitree_z1_dual_arm_stackbox_v2','simulation')
 except ValueError:pass
 else:raise AssertionError(k)
sentinel=object();calls=[]
def original(value):calls.append(value);return sentinel
model=SimpleNamespace(decode_first_stage=original)
os.environ.update(base)
for dataset,stage in [('unitree_z1_dual_arm_stackbox_v2','decision'),('unitree_z1_stackbox','simulation')]:
 assert decode_for_stage(model,None,dataset=dataset,stage=stage,execution_steps=16) is sentinel
os.environ[FLAG]='0'
assert decode_for_stage(model,None,dataset='unitree_z1_dual_arm_stackbox_v2',stage='simulation',execution_steps=16) is sentinel
assert len(calls)==3 and not torch.cuda.is_initialized()
report=dict(passed=True,valid_chunks=4,invalid_chunks=3,dependency_guards=len(DEPS),
            disabled_decision_othercase_native_fallbacks=3,CUDA_context_initialized=False)
Path('vae-protected-tf32-production-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
