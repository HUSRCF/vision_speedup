import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from unifolm_wma.utils.vae_nonfeedback_cache_runtime import FLAG,configured_mode,decode_for_stage,get_report
base={**{'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_FULL_DENOISER_FP16': '0', 'UNIFOLM_BATCH_DECISION_DECODE': '1', 'UNIFOLM_STAGE_TF32': 'decision-cudnn', 'UNIFOLM_STAGE_TF32_ALL_CASES': '0', 'UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS': '0'},'UNIFOLM_WORLD_DECODE_PROTECTED_TF32_CHUNK':'14',FLAG:'fp16_nchw'}
assert not torch.cuda.is_initialized()
assert configured_mode({},'unitree_z1_dual_arm_stackbox_v2','simulation') is None
assert configured_mode(base,'unitree_z1_dual_arm_stackbox_v2','simulation')=='fp16_nchw'
for raw in ['bad','fp16_nhwc','tf32_nhwc']:
 try:configured_mode({**base,FLAG:raw},'unitree_z1_dual_arm_stackbox_v2','simulation')
 except ValueError:pass
 else:raise AssertionError(raw)
for k in [*{'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_FULL_DENOISER_FP16': '0', 'UNIFOLM_BATCH_DECISION_DECODE': '1', 'UNIFOLM_STAGE_TF32': 'decision-cudnn', 'UNIFOLM_STAGE_TF32_ALL_CASES': '0', 'UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS': '0'},'UNIFOLM_WORLD_DECODE_PROTECTED_TF32_CHUNK']:
 wrong={**base,k:'bad'}
 try:configured_mode(wrong,'unitree_z1_dual_arm_stackbox_v2','simulation')
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
assert len(calls)==3 and not torch.cuda.is_initialized() and get_report()['cache_report'] is None
report=dict(passed=True,valid_modes=1,invalid_modes=3,dependency_guards=7,
            disabled_decision_othercase_native_fallbacks=3,CUDA_context_initialized=False,
            no_cache_created_for_disabled_or_otherstage=True)
Path('vae-nonfeedback-cache-production-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
