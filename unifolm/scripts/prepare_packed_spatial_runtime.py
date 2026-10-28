"""Add only opt-in world5x8 packed compensated convolution."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_PACKED_SPATIAL_CONV' not in source
addition = '''    packed_spatial_mode=os.environ.get('UNIFOLM_PACKED_SPATIAL_CONV','0')
    if packed_spatial_mode not in {'0','all','1280'}:
        raise ValueError('Packed spatial scope must be0,all,or1280')
    packed_spatial_count=0
    if packed_spatial_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Packed spatial convolution requires accepted V2 stage policy')
        from .packed_spatial_runtime import install_packed_spatial_convs
        packed_spatial_count=install_packed_spatial_convs(model.model.diffusion_model,packed_spatial_mode)
    print(f'PACKED_SPATIAL_CONV requested={packed_spatial_mode} modules={packed_spatial_count} '
          'scope=world5x8 decision_medium=previous_installed_path BK=64 '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'patch_parts=refreshed_each_timed_call splitK=1_or4 deterministic_reduce=true',flush=True)
'''
source = source.replace(anchor, addition + anchor)
ast.parse(source)
path.write_text(source)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor +
    '    echo "UNIFOLM_PACKED_SPATIAL_CONV=${UNIFOLM_PACKED_SPATIAL_CONV:-0}"\n')
path.write_text(source)
print('PACKED_SPATIAL_RUNTIME_PATCHED', root, flush=True)
