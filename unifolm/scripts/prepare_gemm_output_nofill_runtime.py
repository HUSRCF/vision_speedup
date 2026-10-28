"""Four output allocators only; leave packed inputs and all raw JIT ASTs intact."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
specs=[('triton_half_compensated_linear.py','half_compensated_linear','classic'),
       ('triton_prepacked_input_linear.py','prepacked_input_linear','classic'),
       ('triton_primary_narrow_linear.py','primary_narrow_linear','primary'),
       ('triton_primary_grouped_qkv.py','primary_grouped_qkv','primary')]

def kernels(source):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
        if isinstance(n,ast.FunctionDef) and n.decorator_list}

for filename,name,kind in specs:
    path=root/'src/unifolm_wma/utils'/filename
    source=path.read_text();before=kernels(source)
    assert 'gemm_output_allocation' not in source
    fn=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(fn)==1
    calls=[n for n in ast.walk(fn[0]) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
        and n.func.value.id=='torch' and n.func.attr=='empty']
    assert len(calls)==1,(filename,len(calls))
    n=calls[0].func;lines=source.splitlines(keepends=True)
    start=sum(map(len,lines[:n.lineno-1]))+n.col_offset
    end=sum(map(len,lines[:n.end_lineno-1]))+n.end_col_offset
    assert source[start:end+1]=='torch.empty('
    source=source[:start]+f"empty_gemm_output('{kind}',"+source[end+1:]
    assert source.count('import torch\n')==1
    source=source.replace('import torch\n','import torch\nfrom .gemm_output_allocation import empty as empty_gemm_output\n',1)
    assert kernels(source)==before
    path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'    echo "UNIFOLM_GEMM_OUTPUT_NOFILL=${UNIFOLM_GEMM_OUTPUT_NOFILL:-0}"\n');path.write_text(source)
print('GEMM_OUTPUT_NOFILL_RUNTIME_PATCHED helpers=4 jit_asts=unchanged',root,flush=True)
