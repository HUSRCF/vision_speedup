"""Add opt-in host dispatches; retain every previous raw JIT AST and line."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
specs=[('triton_packed_spatial_conv.py','packed_spatial_conv','_gemm','packed'),
       ('triton_winograd_f2.py','winograd_f2_conv2d','_components_f2','f2')]
def kernels(source):
    return {n.name:(ast.dump(n,include_attributes=False),n.lineno) for n in ast.parse(source).body
            if isinstance(n,ast.FunctionDef) and n.decorator_list}
for filename,function,launch,kind in specs:
    path=root/'src/unifolm_wma/utils'/filename;source=path.read_text();before=kernels(source)
    assert source.count('import torch\n')==1
    source=source.replace('import torch\n','import torch; from .triton_zero_weight_low_spatial import run_packed_components, run_f2_components, residual_counts\n',1)
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==function)
    nodes=[n for n in fn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
           and isinstance(n.value.func,ast.Subscript) and ast.unparse(n.value.func.value)==launch]
    assert len(nodes)==1
    node=nodes[0];lines=source.splitlines(keepends=True)
    guard=("    if not run_packed_components(packed,parts,bias,partial,rows,outputs,inner,height*width,block_m,block_n,splits):\n" if kind=='packed' else
           "    if not run_f2_components(input_parts,weight_parts,product,tiles,outputs,inputs,block_m,block_n,block_k,num_warps,num_stages):\n")
    lines[node.lineno-1:node.end_lineno]=[guard]+['    '+line for line in lines[node.lineno-1:node.end_lineno]]
    source=''.join(lines)
    if kind=='f2':
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='transform_filter_f2')
        ret=next(n for n in fn.body if isinstance(n,ast.Return) and ast.unparse(n.value)=='parts')
        lines=source.splitlines(keepends=True)
        lines.insert(ret.lineno-1,"    residual_counts(parts,'f2',refresh=True)\n")
        source=''.join(lines)
    else:
        anchor='    return pack_half_weight(weight.flatten(1), out=out)'
        assert source.count(anchor)==1
        source=source.replace(anchor,"    parts=pack_half_weight(weight.flatten(1),out=out)\n    residual_counts(parts,'packed',refresh=True)\n    return parts")
    assert kernels(source)==before
    path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
path.write_text(source.replace(anchor,anchor+
    '    echo "UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL=${UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL:-0}"\n'))

source=(root/'scripts/profile_attention_output_nofill_pair.py').read_text()
source=source.replace('UNIFOLM_ATTENTION_OUTPUT_NOFILL','UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL')
source=source.replace('ATTENTION_OUTPUT_NOFILL','ZERO_WEIGHT_LOW_SPATIAL')
ast.parse(source);(root/'scripts/profile_zero_weight_low_spatial_pair.py').write_text(source)
source=(root/'scripts/run_attention_output_nofill_pair.py').read_text()
source=source.replace('attention-output-nofill-graph-pair-','zero-weight-low-spatial-graph-pair-')
source=source.replace("flags['UNIFOLM_ATTENTION_OUTPUT_NOFILL']='0'","flags['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL']='0'")
source=source.replace('scripts/profile_attention_output_nofill_pair.py','scripts/profile_zero_weight_low_spatial_pair.py')
source=source.replace('ATTENTION_OUTPUT_NOFILL_','ZERO_WEIGHT_LOW_SPATIAL_')
ast.parse(source);(root/'scripts/run_zero_weight_low_spatial_pair.py').write_text(source)
print('ZERO_WEIGHT_LOW_SPATIAL_PATCHED original_kernel_asts_and_lines=unchanged predicate=device_outside_Kloop count=first_timed_filter_and_version_refresh',root,flush=True)
