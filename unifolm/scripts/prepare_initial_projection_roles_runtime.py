"""Add a role-isolated experiment; do not enable the broader initial scope."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_DECISION_INITIAL_PRIMARY_ROLE' not in source
addition = '''    initial_primary_role=os.environ.get('UNIFOLM_DECISION_INITIAL_PRIMARY_ROLE','0')
    if initial_primary_role not in {'0','qk','vo'}:
        raise ValueError('Initial primary role must be0,qk,orvo')
    initial_primary_role_count=0
    if initial_primary_role!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or initial_primary_scope not in {'0','ffn'} or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Initial role experiment requires separate V2 stage policy')
        from .decision_initial_projection_roles import install_decision_initial_projection_roles
        initial_primary_role_count=install_decision_initial_projection_roles(
            model.model.diffusion_model,initial_primary_role)
        if initial_primary_role_count!=4:
            raise ValueError('Unexpected initial512 attention projection role count')
    print(f'DECISION_INITIAL_PRIMARY_ROLE role={initial_primary_role} '
          f'modules={initial_primary_role_count} other_roles_world=previous '
          'master_bias_accum_output=fp32 input_conversion=each_timed_call '
          'weights=first_timed_warm',flush=True)
'''
source = source.replace(anchor,addition+anchor)
ast.parse(source)
path.write_text(source)
path = root/'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
source = source.replace(anchor,anchor+
    '    echo "UNIFOLM_DECISION_INITIAL_PRIMARY_ROLE=${UNIFOLM_DECISION_INITIAL_PRIMARY_ROLE:-0}"\n')
path.write_text(source)
print('INITIAL_PROJECTION_ROLES_RUNTIME_PATCHED',root,flush=True)
