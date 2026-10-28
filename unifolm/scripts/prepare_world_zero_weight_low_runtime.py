"""Install an opt-in device-predicated C1280 world linear wrapper."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_WORLD_ZERO_WEIGHT_LOW' not in source
addition = '''    world_zero_weight_low = os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW', '0')
    if world_zero_weight_low not in {'0', '1'}:
        raise ValueError('Invalid zero-weight-residual mode')
    world_zero_weight_low_count = 0
    if world_zero_weight_low == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('World zero residual requires accepted V2 stage policy')
        from .world_zero_weight_low_linear import install_world_zero_weight_low_linears
        world_zero_weight_low_count = install_world_zero_weight_low_linears(model.model.diffusion_model)
        if world_zero_weight_low_count == 0:
            raise ValueError('No eligible world zero-weight-residual linears')
    print(f'WORLD_ZERO_WEIGHT_LOW enabled={world_zero_weight_low} modules={world_zero_weight_low_count} '
          'only=world_M2560_640_N1280_K1280_5120 predicate=device_outside_K_loop '
          'rounding=previous_per_BK64 nonzero_residual=previous3products '
          'input_pack=each_timed_call weights_and_predicate=first_timed_warm '
          'version_refresh=same_storage master_bias_accum_output=fp32', flush=True)
'''
updated = source.replace(anchor, addition + anchor)
ast.parse(updated)
path.write_text(updated)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
path.write_text(source.replace(anchor, anchor +
    '    echo "UNIFOLM_WORLD_ZERO_WEIGHT_LOW=${UNIFOLM_WORLD_ZERO_WEIGHT_LOW:-0}"\n'))
print('WORLD_ZERO_WEIGHT_LOW_RUNTIME_PATCHED', root, flush=True)
