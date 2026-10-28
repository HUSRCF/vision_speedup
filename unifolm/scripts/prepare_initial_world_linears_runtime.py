"""Append world-only BK32 initial512 linears to the coherent accepted runtime."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_INITIAL_WORLD_LINEARS' not in source
assert source.index('initial_decision_count = install_initial_decision_linears(') < source.index(anchor)
addition = '''    initial_world_mode=os.environ.get('UNIFOLM_INITIAL_WORLD_LINEARS','0')
    if initial_world_mode not in {'0','1'}:
        raise ValueError('Initial world linears must be0 or1')
    initial_world_count=0
    if initial_world_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                initial_decision_mode!='1'):
            raise ValueError('Initial world linears require accepted V2 decision wrappers')
        from .initial_world_linears import install_initial_world_linears
        initial_world_count=install_initial_world_linears(model.model.diffusion_model)
    print(f'INITIAL_WORLD_LINEARS requested={initial_world_mode} modules={initial_world_count} '
          'BK=32 scope=world_initial512 decision=previous_installed_path '
          'master_bias_accum_output=fp32 weights=first_timed_warm',flush=True)
'''
source = source.replace(anchor, addition + anchor)
ast.parse(source)
path.write_text(source)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_INITIAL_DECISION_LINEARS=${UNIFOLM_INITIAL_DECISION_LINEARS:-0}"\n'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor +
    '    echo "UNIFOLM_INITIAL_WORLD_LINEARS=${UNIFOLM_INITIAL_WORLD_LINEARS:-0}"\n')
path.write_text(source)
print('INITIAL_WORLD_LINEARS_RUNTIME_PATCHED', root, flush=True)
