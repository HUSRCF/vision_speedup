"""Patch only allocations in four public convolution functions; audit JIT ASTs."""
import ast
import hashlib
from pathlib import Path
import sys

root=Path(sys.argv[1])
specs=[('triton_implicit_temporal_conv.py','implicit_temporal_conv','temporal',1),
       ('triton_compensated_temporal_conv.py','compensated_temporal_conv','temporal',3),
       ('triton_packed_spatial_conv.py','packed_spatial_conv','spatial',3),
       ('triton_winograd_f2.py','winograd_f2_conv2d','f2',3)]

def jit_sources(source):
    tree=ast.parse(source)
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
        if isinstance(n,ast.FunctionDef) and n.decorator_list}

for filename,function,kind,count in specs:
    path=root/'src/unifolm_wma/utils'/filename
    source=path.read_text()
    before=jit_sources(source)
    assert 'conv_buffer_allocation' not in source
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function]
    assert len(functions)==1
    calls=[n for n in ast.walk(functions[0]) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
        and n.func.value.id=='torch' and n.func.attr=='empty']
    assert len(calls)==count,(filename,len(calls),count)
    lines=source.splitlines(keepends=True)
    edits=[]
    for call in calls:
        n=call.func
        start=sum(map(len,lines[:n.lineno-1]))+n.col_offset
        end=sum(map(len,lines[:n.end_lineno-1]))+n.end_col_offset
        assert source[start:end+1]=='torch.empty('
        edits.append((start,end+1,f"empty_conv_buffer('{kind}',"))
    for start,end,replacement in sorted(edits,reverse=True):source=source[:start]+replacement+source[end:]
    assert source.count('import torch\n')==1
    source=source.replace('import torch\n','import torch\nfrom .conv_buffer_allocation import empty as empty_conv_buffer\n',1)
    assert jit_sources(source)==before,'Raw JIT kernel AST changed'
    path.write_text(source)
    print('CONV_BUFFER_ALLOCATOR_PATCH '+filename+' calls='+str(count)+' jit_ast_unchanged=true',flush=True)
path=root/'scripts/team07_run.sh'
source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'    echo "UNIFOLM_CONV_BUFFER_NOFILL=${UNIFOLM_CONV_BUFFER_NOFILL:-0}"\n')
path.write_text(source)
print('CONV_BUFFER_NOFILL_RUNTIME_PATCHED',root,flush=True)
