"""Add an opt-in world C320 output experiment without changing old kernels."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_WORLD_TWO_PRODUCT_OUTPUT' not in source
addition = '''    two_product_protection = os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_OUTPUT', '0')
    if two_product_protection not in {'0', 'input', 'weight'}:
        raise ValueError('Invalid world two-product output protection')
    two_product_count = 0
    if two_product_protection != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0 or
                primary_narrow_mode != 'all' or
                os.environ.get('UNIFOLM_WORLD_C320_PRIMARY_LINEAR', '0') != '0'):
            raise ValueError('Two-product world outputs require accepted V2 stage policy')
        from .world_two_product_output import install_world_two_product_output
        two_product_count = install_world_two_product_output(
            model.model.diffusion_model, two_product_protection)
        if two_product_count == 0:
            raise ValueError('No eligible world two-product output projections')
    print(f'WORLD_TWO_PRODUCT_OUTPUT protect={two_product_protection} modules={two_product_count} '
          'shape=40960x320x320 bk=64 products=2 master_bias_accum_output=fp32 '
          'weights=first_timed_warm dynamic_input=each_timed_call '
          'decision_and_other_world_roles=previous_installed_path', flush=True)
'''
updated = source.replace(anchor, addition + anchor)
ast.parse(updated)
path.write_text(updated)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
path.write_text(source.replace(anchor, anchor +
    '    echo "UNIFOLM_WORLD_TWO_PRODUCT_OUTPUT=${UNIFOLM_WORLD_TWO_PRODUCT_OUTPUT:-0}"\n'))
print('WORLD_TWO_PRODUCT_OUTPUT_RUNTIME_PATCHED', root, flush=True)
