"""Install only the new C320 decision GEGLU arithmetic experiment."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_DECISION_C320_PRIMARY_GEGLU' not in source
addition = '''    c320_primary_geglu_mode=os.environ.get('UNIFOLM_DECISION_C320_PRIMARY_GEGLU','0')
    if c320_primary_geglu_mode not in {'0','both','value','gate'}:
        raise ValueError('Decision C320 GEGLU mode must be0,both,value,orgate')
    c320_primary_geglu_bn=int(os.environ.get('UNIFOLM_DECISION_C320_GEGLU_BN','64'))
    c320_primary_geglu_count=0
    if c320_primary_geglu_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                decision_half_geglu_mode!='1' or world_narrow_geglu_mode!='1'):
            raise ValueError('C320 primary GEGLU requires accepted V2 compensated wrappers')
        from .decision_c320_primary_geglu import install_decision_c320_primary_geglus
        c320_primary_geglu_count=install_decision_c320_primary_geglus(
            model.model.diffusion_model,c320_primary_geglu_mode,c320_primary_geglu_bn)
        if c320_primary_geglu_count!=10:
            raise ValueError('Expected ten C320 GEGLU modules')
    print(f'DECISION_C320_PRIMARY_GEGLU requested={c320_primary_geglu_mode} '
          f'modules={c320_primary_geglu_count} BN={c320_primary_geglu_bn} BK=32 '
          'scope=decision_C320 world=previous_installed_path '
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
    '    echo "UNIFOLM_DECISION_C320_PRIMARY_GEGLU=${UNIFOLM_DECISION_C320_PRIMARY_GEGLU:-0}"\n' +
    '    echo "UNIFOLM_DECISION_C320_GEGLU_BN=${UNIFOLM_DECISION_C320_GEGLU_BN:-64}"\n')
path.write_text(source)
print('DECISION_C320_PRIMARY_GEGLU_RUNTIME_PATCHED', root, flush=True)
