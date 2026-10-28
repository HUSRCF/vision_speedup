"""Opt-in temporal residual predicate after the accepted conv installation."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_WORLD_ZERO_WEIGHT_LOW_TEMPORAL' not in source
addition = '''    temporal_zero_scope = os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_TEMPORAL', '0')
    if temporal_zero_scope not in {'0', 'all', 'implicit', 'packed'}:
        raise ValueError('Invalid temporal residual scope')
    temporal_zero_count = 0
    if temporal_zero_scope != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0 or
                os.environ.get('UNIFOLM_COMPENSATED_TEMPORAL_CONV') != '1' or
                os.environ.get('UNIFOLM_IMPLICIT_TEMPORAL_CONV') != '1'):
            raise ValueError('Temporal residual optimization requires accepted V2 stage/conv policy')
        from .world_zero_weight_low_temporal import install_world_zero_weight_low_temporal
        temporal_zero_count = install_world_zero_weight_low_temporal(model.model.diffusion_model, temporal_zero_scope)
        if temporal_zero_count == 0:
            raise ValueError('No eligible accepted temporal conv wrappers')
    print(f'WORLD_ZERO_WEIGHT_LOW_TEMPORAL scope={temporal_zero_scope} modules={temporal_zero_count} '
          'rounding=previous_per_BK64 predicate=device_outside_K_loop nonzero_residual=previous3products '
          'C320=implicit_g8 C640=implicit_g16 C1280=packed '
          'input_conversion=each_timed_call weights_and_predicate=first_timed_warm '
          'version_refresh=same_storage master_bias_accum_output=fp32 decision=previous', flush=True)
'''
updated = source.replace(anchor, addition + anchor)
ast.parse(updated)
path.write_text(updated)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
path.write_text(source.replace(anchor, anchor +
    '    echo "UNIFOLM_WORLD_ZERO_WEIGHT_LOW_TEMPORAL=${UNIFOLM_WORLD_ZERO_WEIGHT_LOW_TEMPORAL:-0}"\n'))
print('WORLD_ZERO_WEIGHT_LOW_TEMPORAL_RUNTIME_PATCHED', root, flush=True)
