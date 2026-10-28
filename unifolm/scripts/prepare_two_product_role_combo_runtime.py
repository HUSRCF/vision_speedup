"""Combine two separately qualified, disjoint C640 role installers."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
choices = "{'0', 'qk320', 'output640', 'ffn640'}"
call = '''        two_product_role_count = install_world_two_product_roles(
            model.model.diffusion_model, two_product_role, two_product_role_protect)
'''
assert source.count(choices) == 1 and source.count(call) == 1
assert 'output640_ffn640' not in source
source = source.replace(choices, "{'0', 'qk320', 'output640', 'ffn640', 'output640_ffn640'}")
source = source.replace(call, '''        two_product_role_scopes = (('output640', 'ffn640')
            if two_product_role == 'output640_ffn640' else (two_product_role,))
        for two_product_role_scope in two_product_role_scopes:
            two_product_role_count += install_world_two_product_roles(
                model.model.diffusion_model, two_product_role_scope, two_product_role_protect)
''')
ast.parse(source)
path.write_text(source)
print('TWO_PRODUCT_ROLE_COMBO_PATCHED disjoint_output_and_ffn=true kernels=unchanged', root, flush=True)
