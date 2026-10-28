"""Add an independent initial512 decision precision experiment to a fresh root."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_DECISION_INITIAL_PRIMARY' not in source
addition = '''    initial_primary_scope=os.environ.get('UNIFOLM_DECISION_INITIAL_PRIMARY','0')
    if initial_primary_scope not in {'0','projections','ffn','all'}:
        raise ValueError('Invalid initial512 primary scope')
    initial_primary_count=0
    if initial_primary_scope!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Initial512 primary requires accepted V2 stage policy')
        from .decision_initial_primary import install_decision_initial_primary
        initial_primary_count=install_decision_initial_primary(model.model.diffusion_model,
            initial_primary_scope)
        if initial_primary_count!={'projections':8,'ffn':2,'all':10}[initial_primary_scope]:
            raise ValueError('Unexpected initial512 primary module count')
    print(f'DECISION_INITIAL_PRIMARY scope={initial_primary_scope} modules={initial_primary_count} '
          'world=previous master_bias_accum_output=fp32 '
          'static_weight_parts=first_timed_warm input_conversion=each_timed_call',flush=True)
'''
source = source.replace(anchor, addition + anchor)
ast.parse(source)
path.write_text(source)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor +
    '    echo "UNIFOLM_DECISION_INITIAL_PRIMARY=${UNIFOLM_DECISION_INITIAL_PRIMARY:-0}"\n')
path.write_text(source)
print('DECISION_INITIAL_PRIMARY_RUNTIME_PATCHED', root, flush=True)
