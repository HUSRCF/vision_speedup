"""Patch a fresh accepted source copy, keeping production defaults disabled."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_WORLD_C320_BK64_GEGLU' not in source
addition = '''    world_c320_primary_mode=os.environ.get('UNIFOLM_WORLD_C320_BK64_GEGLU','0')
    if world_c320_primary_mode not in {'0','both','value','gate'}:
        raise ValueError('World C320 GEGLU mode must be0,both,value,orgate')
    world_c320_primary_count=0
    if world_c320_primary_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                c320_primary_geglu_mode!='both' or world_narrow_geglu_mode!='1'):
            raise ValueError('World C320 experiment requires accepted V2 wrappers')
        from .world_bk64_c320_geglu import install_world_c320_primary_geglus
        world_c320_primary_count=install_world_c320_primary_geglus(
            model.model.diffusion_model,world_c320_primary_mode,32 if world_c320_primary_mode=='value' else 64)
        if world_c320_primary_count!=10:
            raise ValueError('Expected ten world C320 GEGLU modules')
    print(f'WORLD_BK64_C320_GEGLU requested={world_c320_primary_mode} '
          f'modules={world_c320_primary_count} BN=mode_selected BK=64 '
          'scope=world_C320 decision=previous_installed_path '
          'master_bias_accum_gelu_output=fp32 weights=first_timed_warm '
          'input_parts=refreshed_each_timed_call',flush=True)
'''
source = source.replace(anchor, addition + anchor)
ast.parse(source)
path.write_text(source)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor +
    '    echo "UNIFOLM_WORLD_C320_BK64_GEGLU=${UNIFOLM_WORLD_C320_BK64_GEGLU:-0}"\n')
path.write_text(source)
print('WORLD_BK64_C320_GEGLU_RUNTIME_PATCHED', root, flush=True)
