"""Unique-anchor QKV hook, with an opt-in scope after accepted configuration."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/compensated_qkv.py'
source = path.read_text()
anchor = 'def fused_compensated_qkv(module, value):\n'
assert source.count(anchor) == 1 and '_world_zero_weight_low_qkv_installed' not in source
addition = '''    if getattr(module, '_world_zero_weight_low_qkv_installed', False):
        from .world_zero_weight_low_qkv import try_world_zero_weight_low_qkv
        result = try_world_zero_weight_low_qkv(module, value)
        if result is not None:
            return result
'''
updated = source.replace(anchor, anchor + addition)
ast.parse(updated)
path.write_text(updated)
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV' not in source
addition = '''    world_zero_qkv_scope = os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV', '0')
    if world_zero_qkv_scope not in {'0', 'all', '320', 'wide'}:
        raise ValueError('Invalid world zero-weight-residual QKV scope')
    world_zero_qkv_count = 0
    if world_zero_qkv_scope != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('World QKV residual optimization requires accepted V2 stage policy')
        from .world_zero_weight_low_qkv import install_world_zero_weight_low_qkv
        world_zero_qkv_count = install_world_zero_weight_low_qkv(model.model.diffusion_model, world_zero_qkv_scope)
        if world_zero_qkv_count == 0:
            raise ValueError('No eligible world QKV modules')
    print(f'WORLD_ZERO_WEIGHT_LOW_QKV scope={world_zero_qkv_scope} modules={world_zero_qkv_count} '
          'rounding=previous_per_BK64 predicate=device_outside_K_loop '
          'nonzero_residual=previous3products input_pack=each_timed_call '
          'weights_and_predicate=first_timed_warm version_refresh=same_storage '
          'master_accum_and_3_contiguous_output_planes=fp32 decision=previous', flush=True)
'''
updated = source.replace(anchor, addition + anchor)
ast.parse(updated)
path.write_text(updated)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
path.write_text(source.replace(anchor, anchor +
    '    echo "UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV=${UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV:-0}"\n'))
print('WORLD_ZERO_WEIGHT_LOW_QKV_RUNTIME_PATCHED', root, flush=True)
