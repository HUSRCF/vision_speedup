"""Change only launch grouping in the accepted legacy TF32 GEGLU helper."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/triton_compensated_geglu.py'
source = path.read_text()
def replace(before, after):
    global source
    assert source.count(before) == 1, before[:100]
    source = source.replace(before, after, 1)

replace('import torch\n', 'import os\nimport torch\n')
replace('''                       BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)''', '''                       BN: tl.constexpr, BK: tl.constexpr,
                       GROUP_M: tl.constexpr = 0):
    if GROUP_M:
        pid = tl.program_id(0)
        row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(C, BN)
        group = pid // (GROUP_M * col_tiles)
        first_row = group * GROUP_M
        group_rows = tl.minimum(GROUP_M, row_tiles-first_row)
        local = pid % (GROUP_M * col_tiles)
        row_tile = first_row + local % group_rows
        col_tile = local // group_rows
    else:
        row_tile, col_tile = tl.program_id(0), tl.program_id(1)
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)''')
replace('''                      num_warps=4, num_stages=2):
''', '''                      num_warps=4, num_stages=2, group_m=0):
    if group_m not in {0, 4, 8, 16, 32}:
        raise ValueError('Unsupported GEGLU row tile group')
''')
replace('''    _compensated_geglu[(triton.cdiv(rows, block_m), triton.cdiv(channels, block_n))](
''', '''    row_tiles, col_tiles = triton.cdiv(rows, block_m), triton.cdiv(channels, block_n)
    grid = (row_tiles*col_tiles,) if group_m else (row_tiles, col_tiles)
    _compensated_geglu[grid](
''')
replace('BM=block_m, BN=block_n, BK=32, num_warps=num_warps,',
        'BM=block_m, BN=block_n, BK=32, GROUP_M=group_m, num_warps=num_warps,')
replace('    projection = module.proj\n', '''    projection = module.proj
    group_m = int(os.environ.get('UNIFOLM_COMPENSATED_GEGLU_GROUP_M', '0'))
    if group_m not in {0, 4, 8, 16, 32}:
        raise ValueError('Unsupported GEGLU row tile group')
''')
replace('return compensated_geglu(value, projection.weight, projection.bias)',
        'return compensated_geglu(value, projection.weight, projection.bias, group_m=group_m)')
ast.parse(source)
path.write_text(source)
path = root/'scripts/team07_run.sh'
lines = path.read_text().splitlines(keepends=True)
for i, line in enumerate(lines):
    if 'UNIFOLM_COMPENSATED_GEGLU=' in line and 'echo ' in line:
        lines.insert(i+1, '    echo "UNIFOLM_COMPENSATED_GEGLU_GROUP_M=${UNIFOLM_COMPENSATED_GEGLU_GROUP_M:-0}"\n')
        break
else:
    raise RuntimeError('Missing provenance anchor')
path.write_text(''.join(lines))
print('GROUPED_GEGLU_RUNTIME_PATCHED', root, flush=True)
