"""Opt in after the accepted FP32-only temporal Conv2d wrapper is installed."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py'
source=path.read_text()
before='    temporal_conv2d_count = 0\n'
after='''    compensated_temporal_mode = os.environ.get('UNIFOLM_COMPENSATED_TEMPORAL_CONV', '0')
    if compensated_temporal_mode not in {'0','1'}:
        raise ValueError('UNIFOLM_COMPENSATED_TEMPORAL_CONV must be 0 or 1')
    if compensated_temporal_mode == '1' and temporal_conv_impl != 'fp32-conv2d':
        raise ValueError('Compensated temporal convolution requires accepted fp32-conv2d')
    compensated_temporal_count = 0
''' + before
assert source.count(before)==1
source=source.replace(before,after)
before='                    temporal_conv2d_count += 1\n'
after='''                    if compensated_temporal_mode == '1':
                        from .triton_compensated_temporal_conv import make_compensated_temporal_forward
                        conv.forward = make_compensated_temporal_forward(conv)
                        compensated_temporal_count += 1
''' + before
assert source.count(before)==1
source=source.replace(before,after)
before='    print(f"TEMPORAL_CONV_IMPL requested={temporal_conv_impl} "\n'
after='''    print(f'COMPENSATED_TEMPORAL_CONV requested={compensated_temporal_mode} '
          f'modules={compensated_temporal_count} tf32_stage=native '
          'master_accum_output=fp32 static_weights=first_timed_warm BK=64',flush=True)
''' + before
assert source.count(before)==1
source=source.replace(before,after)
ast.parse(source)
path.write_text(source)
path=root/'scripts/team07_run.sh'
lines=path.read_text().splitlines(keepends=True)
for i,line in enumerate(lines):
    if 'UNIFOLM_TEMPORAL_CONV_IMPL=' in line and 'echo ' in line:
        lines.insert(i+1,'    echo "UNIFOLM_COMPENSATED_TEMPORAL_CONV=${UNIFOLM_COMPENSATED_TEMPORAL_CONV:-0}"\n')
        break
else:
    raise RuntimeError('Missing temporal provenance anchor')
path.write_text(''.join(lines))
print('COMPENSATED_TEMPORAL_RUNTIME_PATCHED',root,flush=True)
