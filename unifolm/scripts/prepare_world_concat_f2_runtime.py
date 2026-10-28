"""Add an opt-in concat scope after the accepted denoiser configuration."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_WORLD_CONCAT_F2' not in source
addition = '''    concat_f2_scope = os.environ.get('UNIFOLM_WORLD_CONCAT_F2', '0')
    if concat_f2_scope not in {'0', 'all', '960', '1280', '1920'}:
        raise ValueError('Invalid world concat F2 scope')
    concat_f2_count = 0
    if concat_f2_scope != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('Concat F2 requires accepted V2 stage policy')
        from .world_concat_f2 import install_world_concat_f2
        concat_f2_count = install_world_concat_f2(model.model.diffusion_model, concat_f2_scope)
        if concat_f2_count == 0:
            raise ValueError('No eligible world concat F2 convolutions')
    print(f'WORLD_CONCAT_F2 scope={concat_f2_scope} modules={concat_f2_count} '
          'only=world_N16_CO640_CI960_1280_1920_H20_W32 '
          'products=3 transforms_accum_master_bias_output=fp32 block_k=64 '
          'input_transform=each_timed_call filter=first_timed_warm '
          'version_refresh=same_storage all_other_paths=previous', flush=True)
'''
updated = source.replace(anchor, addition + anchor)
ast.parse(updated)
path.write_text(updated)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
path.write_text(source.replace(anchor, anchor +
    '    echo "UNIFOLM_WORLD_CONCAT_F2=${UNIFOLM_WORLD_CONCAT_F2:-0}"\n'))
print('WORLD_CONCAT_F2_RUNTIME_PATCHED', root, flush=True)
