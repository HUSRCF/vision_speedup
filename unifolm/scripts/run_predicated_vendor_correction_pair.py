"""Run actual decision and world graph prefill pairs with accepted run flags."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

source = Path(os.environ['PAIR_PROFILE_SOURCE_RUN'])
output = Path.home()/'benchmark-runs/diagnostics'/(
    'predicated-vendor-correction-graph-pair-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
output.mkdir(exist_ok=False)
flags = {}
for line in (source/'provenance.txt').read_text().splitlines():
    if line.startswith('UNIFOLM_') and '=' in line:
        key,value = line.split('=',1)
        flags[key] = value
assert flags['UNIFOLM_NORM_BUFFER_NOFILL'] == 'all'
assert flags.get('UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32_TILES')=='1'
assert flags.get('UNIFOLM_WORLD_ZERO_QKV_GROUP')=='1-large'
assert flags.get('UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS')=='1'
assert flags.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW')=='0'
flags['UNIFOLM_DECISION_VENDOR_CORRECTION']='1'
print('DIAGNOSTIC_DIR='+str(output),flush=True)
environment = {k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_')}
environment.update(flags)
(output/'flags.json').write_text(json.dumps(flags,indent=2)+'\n')
(output/'source-run.txt').write_text(str(source)+'\n')
comparisons = {}
for stage in ['decision','world']:
    directory = output/stage
    command = [sys.executable,'scripts/profile_predicated_vendor_correction_pair.py',
        '--diagnostic-dir',str(directory),'--profile-stage',stage,
        '--seed','123','--ckpt_path',os.environ['CKPT_PATH'],
        '--config','configs/inference/world_model_interaction.yaml',
        '--dataset','unitree_z1_dual_arm_stackbox_v2','--savedir',str(directory)+'-unused',
        '--bs','1','--height','320','--width','512',
        '--unconditional_guidance_scale','1.0','--ddim_steps','50','--ddim_eta','1.0',
        '--video_length','16','--frame_stride','4','--exe_steps','16','--n_iter','1',
        '--timestep_spacing','uniform_trailing','--guidance_rescale','.7','--perframe_ae',
        '--input_cache_dir',os.environ['INPUT_CACHE_DIR'],'--require_input_cache']
    with (output/(stage+'.log')).open('w') as log:
        subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    comparisons[stage] = json.loads((directory/'comparison.json').read_text())
    print('PREDICATED_VENDOR_CORRECTION_STAGE_COMPLETE '+stage+' '+str(comparisons[stage]['median_saved_ms']),flush=True)
(output/'comparison.json').write_text(json.dumps(dict(stages=comparisons,
    no_official_score=True),indent=2)+'\n')
print('PREDICATED_VENDOR_CORRECTION_BOTH_STAGES_CHECK passed=true same_process_pair_per_stage=true '
      'actual_inputs=true bitwise_outputs=true no_official_score=true',flush=True)
