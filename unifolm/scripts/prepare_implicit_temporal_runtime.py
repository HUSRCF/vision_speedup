"""Add opt-in fusion to an exact accepted compensated-temporal source tree."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/triton_compensated_temporal_conv.py'
source = path.read_text()
before = 'import torch\n'
assert source.count(before) == 1
source = source.replace(before, 'import os\n' + before)
before = 'def make_compensated_temporal_forward(module):\n    original = module.forward\n'
after = '''def make_compensated_temporal_forward(module):
    implicit_mode = os.environ.get('UNIFOLM_IMPLICIT_TEMPORAL_CONV', '0')
    if implicit_mode not in {'0', '1'}:
        raise ValueError('UNIFOLM_IMPLICIT_TEMPORAL_CONV must be 0 or 1')
    original = module.forward
'''
assert source.count(before) == 1
source = source.replace(before, after)
before = '''        return compensated_temporal_conv(value,parts,module.bias,
            block_m=64,block_n=64,block_k=64,fused_output=True)
'''
after = '''        if implicit_mode == '1' and value.shape[1] in {320,640}:
            from .triton_implicit_temporal_conv import implicit_temporal_conv
            return implicit_temporal_conv(value,parts,module.bias,
                block_m=64,block_n=64,
                group_m=8 if value.shape[1] == 320 else 16,num_stages=2)
''' + before
assert source.count(before) == 1
source = source.replace(before, after)
ast.parse(source)
path.write_text(source)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
before = '    echo "UNIFOLM_COMPENSATED_TEMPORAL_CONV=${UNIFOLM_COMPENSATED_TEMPORAL_CONV:-0}"\n'
assert source.count(before) == 1
source = source.replace(before, before + '    echo "UNIFOLM_IMPLICIT_TEMPORAL_CONV=${UNIFOLM_IMPLICIT_TEMPORAL_CONV:-0}"\n')
path.write_text(source)
print('IMPLICIT_TEMPORAL_RUNTIME_PATCHED', root, flush=True)
