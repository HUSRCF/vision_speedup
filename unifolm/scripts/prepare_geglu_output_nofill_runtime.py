"""Patch five audited output allocations without changing GEGLU kernel bodies."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
names=['triton_primary_narrow_geglu.py','triton_world_narrow_geglu.py',
       'triton_primary_geglu.py','triton_branch_protected_geglu.py','triton_prepacked_geglu.py']
for name in names:
    path=root/'src/unifolm_wma/utils'/name
    source=path.read_text()
    assert 'geglu_output_allocation' not in source
    matches=[]
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node,ast.Assign) and len(node.targets)==1
                and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='output'
                and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='torch'
                and node.value.func.attr=='empty'):
            matches.append(node.value.func)
    assert len(matches)==1,(name,len(matches))
    node=matches[0]
    lines=source.splitlines(keepends=True)
    start=sum(map(len,lines[:node.lineno-1]))+node.col_offset
    end=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
    assert source[start:end]=='torch.empty'
    source=source[:start]+'empty_geglu_output'+source[end:]
    assert source.count('import torch\n')==1
    source=source.replace('import torch\n','import torch\nfrom .geglu_output_allocation import empty as empty_geglu_output\n',1)
    ast.parse(source)
    path.write_text(source)
path=root/'scripts/team07_run.sh'
source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'    echo "UNIFOLM_GEGLU_OUTPUT_NOFILL=${UNIFOLM_GEGLU_OUTPUT_NOFILL:-0}"\n')
path.write_text(source)
print('GEGLU_OUTPUT_NOFILL_RUNTIME_PATCHED helpers=5 kernel_bodies=unchanged',root,flush=True)
