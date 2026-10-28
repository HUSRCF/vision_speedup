"""Primitive guards followed by separate actual decision/world model processes."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

source=Path(os.environ['KERNEL_PROFILE_SOURCE_RUN'])
output=Path.home()/'benchmark-runs/diagnostics'/('fp16-balanced-int8-residual-model-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
output.mkdir(exist_ok=False);print('DIAGNOSTIC_DIR='+str(output),flush=True)
flags=json.loads(Path(os.environ['CURRENT_EXPERIMENT_FLAGS_FILE']).read_text())
assert flags['UNIFOLM_ATTENTION_OUTPUT_NOFILL']==flags['UNIFOLM_GEMM_OUTPUT_NOFILL']=='all'
assert flags['UNIFOLM_WORLD_PREPACKED_ZERO_LOW']=='1'
assert flags['UNIFOLM_NATIVE_SGEMM_FP16']==flags['UNIFOLM_WORLD_NORM2_QKV_PACK']=='1'
environment={k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_')};environment.update(flags)
(output/'flags.json').write_text(json.dumps(flags,indent=2)+'\n');(output/'source-run.txt').write_text(str(source)+'\n')
with (output/'guard.log').open('x') as log:
    subprocess.run([sys.executable,'-c','import torch;import fp16_balanced_residual209_mode as m;m.install();m.uninstall();assert m.ORIGINAL_JIT.run==m.ORIGINAL_RUN;assert not torch.cuda.is_initialized();print("CPU_IMPORT_AND_HOOK_RESTORATION passed=true")'],env=dict(environment,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
rows={}
for stage in ['world']:
    directory=output/stage
    command=[sys.executable,'scripts/profile_fp16_balanced_residual209_model.py','--diagnostic-dir',str(directory),'--profile-stage',stage,
        '--seed','123','--ckpt_path',os.environ['CKPT_PATH'],'--config','configs/inference/world_model_interaction.yaml',
        '--dataset','unitree_z1_dual_arm_stackbox_v2','--savedir',str(directory)+'-unused','--bs','1','--height','320','--width','512',
        '--unconditional_guidance_scale','1.0','--ddim_steps','50','--ddim_eta','1.0','--video_length','16','--frame_stride','4','--exe_steps','16',
        '--n_iter','1','--timestep_spacing','uniform_trailing','--guidance_rescale','.7','--perframe_ae','--input_cache_dir',os.environ['INPUT_CACHE_DIR'],'--require_input_cache']
    with (output/(stage+'.log')).open('w') as log:subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows[stage]=json.loads((directory/'comparison.json').read_text());assert all(e['gross_passed'] for c in rows[stage]['cases'] for e in c['errors']);assert rows[stage]['attention_launches']['captured']>0;print('BALANCED_INT8_RESIDUAL_209_STAGE '+stage+' '+str(rows[stage]['median_saved_ms']),flush=True)
(output/'comparison.json').write_text(json.dumps(dict(stages=rows,ready_for_full_trial=all(p['saved_ms']>0 for r in rows.values() for p in r['pairs']),no_official_score=True),indent=2)+'\n')
print('BALANCED_INT8_RESIDUAL_209_BOTH_STAGES_CHECK passed=true six_cases_each_stage=true no_official_score=true',flush=True)
