"""Change only the allocator of completely written High/Low split buffers."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/triton_half_compensated_linear.py'
source=path.read_text()
anchor='        torch.empty_like(weight, dtype=torch.float16) for _ in range(2))'
assert source.count(anchor)==1 and 'half_pack_allocation' not in source
source=source.replace('import torch\n','import torch\nfrom .half_pack_allocation import empty_like as empty_half_pack\n',1)
source=source.replace(anchor,'        empty_half_pack(weight) for _ in range(2))')
ast.parse(source)
path.write_text(source)
path=root/'scripts/team07_run.sh'
source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'    echo "UNIFOLM_HALF_PACK_NOFILL=${UNIFOLM_HALF_PACK_NOFILL:-0}"\n')
path.write_text(source)
print('HALF_PACK_NOFILL_RUNTIME_PATCHED',root,flush=True)
