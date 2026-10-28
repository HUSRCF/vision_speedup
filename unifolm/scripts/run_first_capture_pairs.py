"""Four fresh processes, baseline/candidate/candidate/baseline on one GPU."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

source = Path(os.environ['PAIR_PROFILE_SOURCE_RUN'])
output = Path.home()/'benchmark-runs/diagnostics'/('first-capture-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
output.mkdir(exist_ok=False)
flags = dict(line.split('=', 1) for line in (source/'provenance.txt').read_text().splitlines()
             if line.startswith('UNIFOLM_') and '=' in line)
assert flags['UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS'] == '1'
flags['UNIFOLM_NORM_TEMPORAL_ZERO_PIPELINE'] = '0'
environment = {k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_')}
(output/'flags.json').write_text(json.dumps(flags, indent=2)+'\n')
(output/'source-run.txt').write_text(str(source)+'\n')
print('DIAGNOSTIC_DIR='+str(output), flush=True)
trials = []
for index, mode in enumerate(['0', '1', '1', '0']):
    label = str(index)+'-'+('baseline' if mode == '0' else 'candidate')
    directory = output/label
    env = dict(environment, **flags)
    assert env['UNIFOLM_DECISION_PRIMARY_HIDDEN_PARTS']=='1'
    assert env['UNIFOLM_WORLD_ZERO_QKV_STAGES']=='3-large'
    env['UNIFOLM_WORLD_FFN1280_ZERO_STAGES'] = '2' if mode=='0' else '3'
    command = [sys.executable, 'scripts/profile_first_capture.py', '--diagnostic-dir', str(directory),
        '--seed','123','--ckpt_path',os.environ['CKPT_PATH'],
        '--config','configs/inference/world_model_interaction.yaml',
        '--dataset','unitree_z1_dual_arm_stackbox_v2','--savedir',str(directory)+'-unused',
        '--bs','1','--height','320','--width','512',
        '--unconditional_guidance_scale','1.0','--ddim_steps','50','--ddim_eta','1.0',
        '--video_length','16','--frame_stride','4','--exe_steps','16','--n_iter','1',
        '--timestep_spacing','uniform_trailing','--guidance_rescale','.7','--perframe_ae',
        '--input_cache_dir',os.environ['INPUT_CACHE_DIR'],'--require_input_cache']
    with (output/(label+'.log')).open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    stages = json.loads((directory/'capture.json').read_text())
    assert [s['stage'] for s in stages] == ['decision', 'world']
    assert all(len(s['forwards']) == 4 and sum(f['graph_capturing'] for f in s['forwards']) == 1 for s in stages)
    if trials:
        assert [s['outputs'] for s in stages] == [s['outputs'] for s in trials[0]['stages']]
        def geometry(rows):
            return [[{k:v for k,v in r.items() if k!='num_stages'} for r in stage['FFN1280_launches']] for stage in rows]
        assert geometry(stages)==geometry(trials[0]['stages'])
    trials.append(dict(label=label, ffn1280_stage_mode=mode, stages=stages))
    print('FIRST_CAPTURE_TRIAL_COMPLETE '+label, flush=True)
(output/'comparison.json').write_text(json.dumps(dict(trials=trials, all_stage_output_SHA_identical=True,
    order=['baseline','candidate','candidate','baseline'], fresh_process_each_trial=True,
    instrumentation_changes_synchronization=True, no_official_score=True), indent=2)+'\n')
print('FIRST_CAPTURE_PAIR_CHECK passed=true trials=4 both_stage_output_SHA_identical=true no_official_score=true', flush=True)
