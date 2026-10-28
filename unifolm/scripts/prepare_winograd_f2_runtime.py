"""Add a separate F2 opt-in without replacing other accepted runtime code."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
before = '    fused_temporal_attention = os.environ.get("UNIFOLM_FUSED_TEMPORAL_ATTENTION", "0")\n'
after = '''    winograd_f2 = os.environ.get('UNIFOLM_WINOGRAD_F2', '0')
    if winograd_f2 not in {'0', 'high', 'medium', 'deep'}:
        raise ValueError('UNIFOLM_WINOGRAD_F2 must be 0, high, medium, or deep')
    winograd_f2_count = 0
    if winograd_f2 != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or mode != 'native' or
                winograd_conv != '0' or compensated_spatial != '0'):
            raise ValueError('F2 requires V2 FP32 native convolutions and no other spatial replacement')
        from unifolm_wma.utils.triton_winograd_f2 import install_f2_convolutions
        winograd_f2_count = install_f2_convolutions(model.model.diffusion_model, scope=winograd_f2)
    print(f'WINOGRAD_F2 requested={winograd_f2} modules={winograd_f2_count} '
          f'spatial_scope={winograd_f2} master=fp32 transforms=fp32 '
          'component_accum=fp32 filter_cache=first_timed_warm', flush=True)
''' + before
assert source.count(before) == 1
source = source.replace(before, after)
ast.parse(source)
path.write_text(source)
path = root/'scripts/team07_run.sh'
lines = path.read_text().splitlines(keepends=True)
for i, line in enumerate(lines):
    if 'UNIFOLM_WINOGRAD_CONV=' in line and 'echo ' in line:
        lines.insert(i+1, '    echo "UNIFOLM_WINOGRAD_F2=${UNIFOLM_WINOGRAD_F2:-0}"\n')
        break
else:
    raise RuntimeError('Missing provenance anchor')
path.write_text(''.join(lines))
print('WINOGRAD_F2_RUNTIME_PATCHED', root, flush=True)
