"""Patch only the audited, fully-written custom normalization allocations."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
pending = []
for name, kind, expected in [('triton_split_group_norm.py', 'group', (3, 0)),
                             ('triton_residual_layer_norm.py', 'residual', (0, 2))]:
    path = root / 'src/unifolm_wma/utils' / name
    source = path.read_text()
    assert 'norm_buffer_allocation' not in source
    assert (source.count('torch.empty('), source.count('torch.empty_like(')) == expected
    assert source.count('import torch\n') == 1
    source = source.replace('import torch\n',
        'import torch\nfrom . import norm_buffer_allocation as _norm_allocation\n')
    source = source.replace('torch.empty(', f'_norm_allocation.empty({kind!r}, ')
    source = source.replace('torch.empty_like(', f'_norm_allocation.empty_like({kind!r}, ')
    ast.parse(source)
    pending.append((path, source))
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1 and 'UNIFOLM_NORM_BUFFER_NOFILL' not in source
source = source.replace(anchor, anchor +
    '    echo "UNIFOLM_NORM_BUFFER_NOFILL=${UNIFOLM_NORM_BUFFER_NOFILL:-0}"\n')
pending.append((path, source))
for path, source in pending:
    path.write_text(source)
print('NORM_BUFFER_NOFILL_RUNTIME_PATCHED', root)
