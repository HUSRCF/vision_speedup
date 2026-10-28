"""Add one named, decision-only precision experiment to an accepted source."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_DECISION_PRIMARY_NARROW' not in source
addition = '''    primary_narrow_mode=os.environ.get('UNIFOLM_DECISION_PRIMARY_NARROW','0')
    if primary_narrow_mode not in {'0','qkv','output','all'}:
        raise ValueError('Decision primary narrow scope must be0,qkv,output,orall')
    primary_narrow_count=0
    if primary_narrow_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Primary C320 projections require accepted V2 decision policy')
        from .decision_primary_narrow import install_decision_primary_narrow
        primary_narrow_count=install_decision_primary_narrow(model.model.diffusion_model,primary_narrow_mode)
        if primary_narrow_count==0:
            raise ValueError('No eligible C320 projections found')
    print(f'DECISION_PRIMARY_NARROW requested={primary_narrow_mode} modules={primary_narrow_count} '
          'shape=40960x320x320 world=previous_installed_path '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'input_cast_and_epilogue=fused_each_timed_call',flush=True)
'''
source = source.replace(anchor, addition + anchor)
ast.parse(source)
path.write_text(source)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor +
    '    echo "UNIFOLM_DECISION_PRIMARY_NARROW=${UNIFOLM_DECISION_PRIMARY_NARROW:-0}"\n')
path.write_text(source)
print('DECISION_PRIMARY_NARROW_RUNTIME_PATCHED', root, flush=True)
