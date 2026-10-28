"""Insert the initial512 QKV hook without changing the accepted attention source."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/compensated_qkv.py'
source = path.read_text()
anchor = 'def fused_compensated_qkv(module, value):\n'
assert source.count(anchor) == 1 and '_initial_bk32_qkv_installed' not in source
addition = '''    if getattr(module, '_initial_bk32_qkv_installed', False):
        from .initial_bk32_qkv import try_initial_bk32_qkv
        result = try_initial_bk32_qkv(module, value)
        if result is not None:
            return result
'''
updated = source.replace(anchor, anchor+addition)
ast.parse(updated)
path.write_text(updated)
path = root/'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_INITIAL_BK32_QKV' not in source
addition = '''    initial_qkv_mode = os.environ.get('UNIFOLM_INITIAL_BK32_QKV', '0')
    if initial_qkv_mode not in {'0', '1'}:
        raise ValueError('Invalid initial512 BK32 QKV mode')
    initial_qkv_count = 0
    if initial_qkv_mode == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0 or
                os.environ.get('UNIFOLM_INITIAL_WORLD_LINEARS') != '1'):
            raise ValueError('Initial512 QKV requires accepted V2 stage policy')
        from .initial_bk32_qkv import install_initial_bk32_qkv
        initial_qkv_count = install_initial_bk32_qkv(model.model.diffusion_model)
        if initial_qkv_count != 2:
            raise ValueError('Expected exactly two initial512 self-attention modules')
    print(f'INITIAL_BK32_QKV enabled={initial_qkv_mode} modules={initial_qkv_count} '
          'only=world_initial512_M40960 rounding=previous_per_BK32 '
          'input_pack=once_per_QKV_call static_weights_and_predicate=first_timed_warm '
          'nonzero_residual=previous3products version_refresh=same_storage '
          'master_accum_output=fp32 layout=3_contiguous_planes decision=previous', flush=True)
'''
updated = source.replace(anchor, addition+anchor)
ast.parse(updated)
path.write_text(updated)
path = root/'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
path.write_text(source.replace(anchor, anchor+
    '    echo "UNIFOLM_INITIAL_BK32_QKV=${UNIFOLM_INITIAL_BK32_QKV:-0}"\n'))
print('INITIAL_BK32_QKV_RUNTIME_PATCHED', root, flush=True)
