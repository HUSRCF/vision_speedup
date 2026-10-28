"""Install one separate precision scope without changing existing JIT kernels."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_WORLD_TWO_PRODUCT_ROLE' not in source
addition = '''    two_product_role = os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE', '0')
    two_product_role_protect = os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT', 'input')
    if (two_product_role not in {'0', 'qk320', 'output640', 'ffn640'} or
            two_product_role_protect not in {'input', 'weight'}):
        raise ValueError('Invalid world two-product role/protection')
    two_product_role_count = 0
    if two_product_role != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0 or
                os.environ.get('UNIFOLM_WORLD_C320_PRIMARY_LINEAR', '0') != '0'):
            raise ValueError('World two-product role requires accepted V2 stage policy')
        from .world_two_product_roles import install_world_two_product_roles
        two_product_role_count = install_world_two_product_roles(
            model.model.diffusion_model, two_product_role, two_product_role_protect)
        if two_product_role_count == 0:
            raise ValueError('No eligible world two-product roles found')
    print(f'WORLD_TWO_PRODUCT_ROLE scope={two_product_role} protect={two_product_role_protect} '
          f'modules={two_product_role_count} bk=64 products=2 master_bias_accum_output=fp32 '
          'weights=first_timed_warm version_refresh=same_storage '
          'dynamic_input=each_timed_call decision_and_other_roles=previous', flush=True)
'''
updated = source.replace(anchor, addition + anchor)
ast.parse(updated)
path.write_text(updated)
path = root/'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
path.write_text(source.replace(anchor, anchor +
    '    echo "UNIFOLM_WORLD_TWO_PRODUCT_ROLE=${UNIFOLM_WORLD_TWO_PRODUCT_ROLE:-0}"\n'
    '    echo "UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT=${UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT:-input}"\n'))
print('WORLD_TWO_PRODUCT_ROLES_RUNTIME_PATCHED', root, flush=True)
