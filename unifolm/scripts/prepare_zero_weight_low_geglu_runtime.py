"""Preserve original kernel ASTs/lines and hook only host launches/static refresh."""
import ast
from pathlib import Path
import re
import sys
root=Path(sys.argv[1])
def kernels(text):
    return {n.name:(ast.dump(n,include_attributes=False),n.lineno) for n in ast.parse(text).body
            if isinstance(n,ast.FunctionDef) and n.decorator_list}
specs=[('triton_world_narrow_geglu.py','primary_narrow_geglu','_geglu',
    'inputs,parts,bias,output,v,g,rows,channels,inner,64,block_n,16,mode in {"value","both"},mode in {"gate","both"},debug_branches,block_k,4,2'),
    ('triton_branch_protected_geglu.py','branch_protected_geglu','_geglu',
    'inputs,parts,bias,output,v,g,rows,channels,inner,64,block_n,group_m,mode in {"value","both"},mode in {"gate","both"},debug_branches,block_k,4,2'),
    ('triton_prepacked_geglu.py','prepacked_geglu','_prepacked_geglu',
    'packed,parts,bias,output,output,output,rows,channels,inner,block_m,block_n,group_m,False,False,False,64,num_warps,num_stages')]
for filename,function,launch,args in specs:
    path=root/'src/unifolm_wma/utils'/filename;source=path.read_text();before=kernels(source)
    assert source.count('import torch\n')==1
    source=source.replace('import torch\n','import torch; from .triton_zero_weight_low_geglu import run as run_zero_geglu, residual_count as refresh_geglu_count\n',1)
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==function)
    node=next(n for n in fn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
              and isinstance(n.value.func,ast.Subscript) and ast.unparse(n.value.func.value)==launch)
    lines=source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=['    if not run_zero_geglu('+args+'):\n']+[
        '    '+line for line in lines[node.lineno-1:node.end_lineno]]
    source=''.join(lines)
    if filename=='triton_prepacked_geglu.py':
        matches=list(re.finditer(r'^( +)parts\s*=\s*pack_half_weight\(weight,\s*out=reuse\)\n',source,re.M))
        assert len(matches)==1
        match=matches[0];source=source[:match.end()]+match.group(1)+'refresh_geglu_count(parts,refresh=True)\n'+source[match.end():]
    assert kernels(source)==before
    path.write_text(source)
for filename in ['world_bk64_c320_geglu.py','world_wide_gate_geglu.py']:
    path=root/'src/unifolm_wma/utils'/filename;source=path.read_text()
    assert source.count('import torch\n')==1
    source=source.replace('import torch\n','import torch\nfrom .triton_zero_weight_low_geglu import residual_count as refresh_geglu_count\n',1)
    matches=list(re.finditer(r'^( +)parts\s*=\s*pack_half_weight\(weight,\s*out=reuse\)\n',source,re.M))
    assert len(matches)==1
    match=matches[0];source=source[:match.end()]+match.group(1)+'refresh_geglu_count(parts,refresh=True)\n'+source[match.end():]
    ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
path.write_text(source.replace(anchor,anchor+
    '    echo "UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU=${UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU:-0}"\n'))
source=(root/'scripts/profile_attention_output_nofill_pair.py').read_text()
source=source.replace('UNIFOLM_ATTENTION_OUTPUT_NOFILL','UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU')
source=source.replace('ATTENTION_OUTPUT_NOFILL','ZERO_WEIGHT_LOW_GEGLU')
ast.parse(source);(root/'scripts/profile_zero_weight_low_geglu_pair.py').write_text(source)
source=(root/'scripts/run_attention_output_nofill_pair.py').read_text()
source=source.replace('attention-output-nofill-graph-pair-','zero-weight-low-geglu-graph-pair-')
source=source.replace("strftime('%Y%m%dT%H%M%SZ')","strftime('%Y%m%dT%H%M%S%fZ')")
source=source.replace("flags['UNIFOLM_ATTENTION_OUTPUT_NOFILL']='0'","flags['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU']='0'")
source=source.replace('scripts/profile_attention_output_nofill_pair.py','scripts/profile_zero_weight_low_geglu_pair.py')
source=source.replace('ATTENTION_OUTPUT_NOFILL_','ZERO_WEIGHT_LOW_GEGLU_')
ast.parse(source);(root/'scripts/run_zero_weight_low_geglu_pair.py').write_text(source)
print('ZERO_WEIGHT_LOW_GEGLU_PATCHED old_jit_asts_and_lines=unchanged original_BK_BM_BN_group_primary_and_debug=preserved static_counts=first_timed_pack_and_refresh',root,flush=True)
