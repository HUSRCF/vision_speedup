import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
import unifolm_wma.utils.vae_nonfeedback_overlap_runtime as runtime
from unifolm_wma.utils.vae_nonfeedback_cache_runtime import get_report as parent_report
base={**{'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_FULL_DENOISER_FP16': '0', 'UNIFOLM_BATCH_DECISION_DECODE': '1', 'UNIFOLM_STAGE_TF32': 'decision-cudnn', 'UNIFOLM_STAGE_TF32_ALL_CASES': '0', 'UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS': '0'},'UNIFOLM_WORLD_DECODE_PROTECTED_TF32_CHUNK':'14','UNIFOLM_WORLD_DECODE_NONFEEDBACK_CACHE':'fp16_nchw',runtime.FLAG:'1'}
assert not torch.cuda.is_initialized()
assert not runtime.configured_overlap({},'unitree_z1_dual_arm_stackbox_v2','simulation')
assert runtime.configured_overlap(base,'unitree_z1_dual_arm_stackbox_v2','simulation')
for raw in ['bad','true','2']:
 try:runtime.configured_overlap({**base,runtime.FLAG:raw},'unitree_z1_dual_arm_stackbox_v2','simulation')
 except ValueError:pass
 else:raise AssertionError(raw)
for k in [*{'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_FULL_DENOISER_FP16': '0', 'UNIFOLM_BATCH_DECISION_DECODE': '1', 'UNIFOLM_STAGE_TF32': 'decision-cudnn', 'UNIFOLM_STAGE_TF32_ALL_CASES': '0', 'UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS': '0'},'UNIFOLM_WORLD_DECODE_PROTECTED_TF32_CHUNK','UNIFOLM_WORLD_DECODE_NONFEEDBACK_CACHE']:
 try:runtime.configured_overlap({**base,k:'bad'},'unitree_z1_dual_arm_stackbox_v2','simulation')
 except ValueError:pass
 else:raise AssertionError(k)
sentinel=object();native_calls=[]
def original(value):native_calls.append(value);return sentinel
model=SimpleNamespace(decode_first_stage=original)
os.environ.update(base)
for dataset,stage in [('unitree_z1_dual_arm_stackbox_v2','decision'),('unitree_z1_stackbox','simulation')]:
 assert runtime.decode_for_stage(model,None,dataset=dataset,stage=stage,execution_steps=16) is sentinel
old=runtime.base_decode;parent_calls=[]
def parent(*args,**kwargs):parent_calls.append(kwargs);return sentinel
runtime.base_decode=parent;os.environ[runtime.FLAG]='0'
try:assert runtime.decode_for_stage(model,None,dataset='unitree_z1_dual_arm_stackbox_v2',stage='simulation',execution_steps=16) is sentinel
finally:runtime.base_decode=old
assert len(native_calls)==2 and len(parent_calls)==1
assert not torch.cuda.is_initialized() and runtime.get_report()['cache_report'] is None and parent_report()['cache_report'] is None
report=dict(passed=True,valid_flags=1,invalid_flags=3,dependency_guards=8,scope_native_fallbacks=2,disabled_parent_cache_fallback=1,CUDA_context_initialized=False,no_cache_or_stream_created=True)
Path('vae-nonfeedback-overlap-production-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
