"""Add only the world C320 cached-weight three-product experiment."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_WORLD_PREPACKED_NARROW_GEGLU' not in source
assert source.index('decision_half_geglu_count = install_decision_half_geglus(') < source.index(anchor)
addition = '''    world_narrow_geglu_mode=os.environ.get('UNIFOLM_WORLD_PREPACKED_NARROW_GEGLU','0')
    if world_narrow_geglu_mode not in {'0','1'}:
        raise ValueError('World narrow GEGLU must be0 or1')
    world_narrow_geglu_tile=tuple(int(v) for v in os.environ.get('UNIFOLM_WORLD_NARROW_GEGLU_TILE','64,32,16').split(','))
    world_narrow_geglu_count=0
    if world_narrow_geglu_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                decision_half_geglu_mode!='1'):
            raise ValueError('World narrow GEGLU requires accepted V2 decision stage policy')
        from .world_prepacked_narrow_geglu import install_world_prepacked_narrow_geglus
        world_narrow_geglu_count=install_world_prepacked_narrow_geglus(
            model.model.diffusion_model,world_narrow_geglu_tile)
    print(f'WORLD_PREPACKED_NARROW_GEGLU requested={world_narrow_geglu_mode} '
          f'modules={world_narrow_geglu_count} tile={world_narrow_geglu_tile} BK=64 '
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
    '    echo "UNIFOLM_WORLD_PREPACKED_NARROW_GEGLU=${UNIFOLM_WORLD_PREPACKED_NARROW_GEGLU:-0}"\n' +
    '    echo "UNIFOLM_WORLD_NARROW_GEGLU_TILE=${UNIFOLM_WORLD_NARROW_GEGLU_TILE:-64,32,16}"\n')
path.write_text(source)
print('WORLD_PREPACKED_NARROW_GEGLU_RUNTIME_PATCHED', root, flush=True)
