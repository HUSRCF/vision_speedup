"""Profile the exact accepted source's third warmed graph call; no score."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

source = Path(os.environ['KERNEL_PROFILE_SOURCE_RUN'])
stage = 'world'
assert stage in {'world', 'decision'}
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output = Path.home() / 'benchmark-runs/diagnostics' / ('current-kernels-' + stage + '-' + stamp)
flags=json.loads(Path(os.environ['CURRENT_EXPERIMENT_FLAGS_FILE']).read_text())
assert flags['UNIFOLM_WORLD_PERSISTENT_ZERO_LOW']=='1'
assert flags['UNIFOLM_CUDA_GRAPH'] == '1'
environment = {k: v for k, v in os.environ.items() if not k.startswith('UNIFOLM_')}
environment.update(flags)
print('DIAGNOSTIC_DIR=' + str(output), flush=True)
command = [sys.executable, 'scripts/profile_fp16_scalar_residual209.py', '--diagnostic-dir', str(output),
    '--require-cudnn-tf32', 'off' if stage == 'world' else 'on',
    '--seed', '123', '--ckpt_path', os.environ['CKPT_PATH'],
    '--config', 'configs/inference/world_model_interaction.yaml',
    '--dataset', 'unitree_z1_dual_arm_stackbox_v2', '--savedir', str(output) + '-unused',
    '--bs', '1', '--height', '320', '--width', '512',
    '--unconditional_guidance_scale', '1.0', '--ddim_steps', '50', '--ddim_eta', '1.0',
    '--video_length', '16', '--frame_stride', '4', '--exe_steps', '16', '--n_iter', '1',
    '--timestep_spacing', 'uniform_trailing', '--guidance_rescale', '.7', '--perframe_ae',
    '--input_cache_dir', os.environ['INPUT_CACHE_DIR'], '--require_input_cache']
allocation_audit=os.environ.get('KERNEL_ALLOCATION_AUDIT','0')=='1'
filter_low_audit=os.environ.get('KERNEL_FILTER_LOW_AUDIT','0')=='1'
weight_cache_audit=os.environ.get('KERNEL_WEIGHT_CACHE_AUDIT','0')=='1'
linear_low_audit=os.environ.get('KERNEL_LINEAR_LOW_AUDIT','0')=='1'
inline_low_audit=os.environ.get('KERNEL_INLINE_LOW_AUDIT','0')=='1'
attention_parts_audit=os.environ.get('KERNEL_ATTENTION_PARTS_AUDIT','0')=='1'
if allocation_audit:command.append('--allocation-audit')
if filter_low_audit:command.append('--filter-low-audit')
if weight_cache_audit:command.append('--weight-cache-audit')
if linear_low_audit:command.append('--linear-low-audit')
if inline_low_audit:command.append('--inline-low-audit')
if attention_parts_audit:command.append('--attention-parts-audit')
subprocess.run(command, env=environment, check=True)
(output / 'source-run.txt').write_text(str(source) + '\n')
(output / 'flags.json').write_text(json.dumps(flags, indent=2) + '\n')
proof=json.loads((output/'replay-proof.json').read_text());assert all(proof.values())
print('FP16_SCALAR_RESIDUAL_209_CHECK passed=true',flush=True)
