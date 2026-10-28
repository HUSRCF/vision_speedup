import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from unifolm_wma.models.diffusion_head.conv1d_components import Conv1dBlock
from unifolm_wma.utils.head_norm_mish_runtime import FLAG,configured,configure_head_fusion
assert not torch.cuda.is_initialized()
dataset='unitree_z1_dual_arm_stackbox_v2'
assert not configured({},'native',dataset)
assert configured({FLAG:'1'},'native',dataset)
assert not configured({FLAG:'1'},'native','unitree_z1_stackbox')
for raw in ['bad','true','2']:
 try:configured({FLAG:raw},'native',dataset)
 except ValueError:pass
 else:raise AssertionError(raw)
for environment,mode in [({FLAG:'1'},'fp16'),({FLAG:'1','UNIFOLM_SIM_HEAD_CONV1D':'gemm'},'native')]:
 try:configured(environment,mode,dataset)
 except ValueError:pass
 else:raise AssertionError('dependency')
heads=torch.nn.ModuleDict({k:torch.nn.ModuleList([Conv1dBlock(8,8,1) for _ in range(37)]) for k in ['action','state']}).eval()
diffusion=SimpleNamespace(action_unet=heads['action'],state_unet=heads['state'])
model=SimpleNamespace(model=SimpleNamespace(diffusion_model=diffusion))
keys=list(heads.state_dict());pointers=[p.data_ptr() for p in heads.parameters()]
saved=[p.detach().clone() for p in heads.parameters()];value=torch.randn(1,8,4)
with torch.no_grad():expected=[block(value) for head in heads.values() for block in head]
os.environ[FLAG]='1';os.environ['UNIFOLM_SIM_HEAD_CONV1D']='native'
controller=configure_head_fusion(model,mode='native',dataset=dataset)
with torch.no_grad():actual=[block(value) for head in heads.values() for block in head]
assert all(torch.equal(a.view(torch.int32),b.view(torch.int32)) for a,b in zip(actual,expected))
assert sum(controller.fallbacks.values())==74 and not controller.hits
assert keys==list(heads.state_dict()) and pointers==[p.data_ptr() for p in heads.parameters()]
assert all(torch.equal(p,s) for p,s in zip(heads.parameters(),saved))
controller.restore()
assert not torch.cuda.is_initialized()
report=dict(passed=True,invalid_flags=3,dependency_guards=2,other_dataset_disabled=True,
    CPU_native_fallbacks=74,state_keys_pointers_values_unchanged=True,CUDA_context_initialized=False)
Path('head-norm-mish-production-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
