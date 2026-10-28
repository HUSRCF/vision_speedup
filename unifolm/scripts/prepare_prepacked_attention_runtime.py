"""Route large streaming attention through a once-per-call dynamic K/V pack."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/triton_compensated_attention.py'
source=path.read_text()
def kernels(text):
    return {n.name:(ast.dump(n,include_attributes=False),n.lineno) for n in ast.parse(text).body
            if isinstance(n,ast.FunctionDef) and n.decorator_list}
before=kernels(source)
assert source.count('import torch;')==1
source=source.replace('import torch;','import os; import torch;',1)
function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='compensated_attention')
node=next(n for n in function.body if isinstance(n,ast.Assign)
          and ast.unparse(n.targets[0])=='output')
lines=source.splitlines(keepends=True)
dispatch="""    mode=os.environ.get('UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV','0')
    if mode not in {'0','kv'}:
        raise ValueError('Invalid spatial attention K/V packing scope')
    if mode=='kv' and rows>=160 and key.shape[1]>=160:
        from .triton_prepacked_compensated_attention import prepacked_compensated_attention
        return prepacked_compensated_attention(query,key,value,scale=scale,
            block_m=block_m,block_n=block_n,num_warps=num_warps,return_kernel=return_kernel)
"""
lines.insert(node.lineno-1,dispatch)
source=''.join(lines)
assert kernels(source)==before
path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
path.write_text(source.replace(anchor,anchor+
    '    echo "UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV=${UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV:-0}"\n'))

source=(root/'scripts/profile_attention_output_nofill_pair.py').read_text()
source=source.replace('UNIFOLM_ATTENTION_OUTPUT_NOFILL','UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV')
source=source.replace("[('baseline','0'),('candidate','all')]","[('baseline','0'),('candidate','kv')]")
source=source.replace('ATTENTION_OUTPUT_NOFILL','PREPACKED_ATTENTION')
ast.parse(source);(root/'scripts/profile_prepacked_attention_pair.py').write_text(source)
source=(root/'scripts/run_attention_output_nofill_pair.py').read_text()
source=source.replace('attention-output-nofill-graph-pair-','prepacked-attention-graph-pair-')
source=source.replace("flags['UNIFOLM_ATTENTION_OUTPUT_NOFILL']='0'",
    "flags['UNIFOLM_ATTENTION_OUTPUT_NOFILL']='all'\nflags['UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV']='0'")
source=source.replace('scripts/profile_attention_output_nofill_pair.py','scripts/profile_prepacked_attention_pair.py')
source=source.replace('ATTENTION_OUTPUT_NOFILL_','PREPACKED_ATTENTION_')
ast.parse(source);(root/'scripts/run_prepacked_attention_pair.py').write_text(source)

source=(root/'scripts/benchmark_prepacked_compensated_attention.py').read_text()
anchor="""        baseline=lambda x:compensated_attention(x,k,v,**kwargs)
        candidate=lambda x:prepacked_compensated_attention(x,k,v,**kwargs)
"""
assert source.count(anchor)==1
source=source.replace(anchor,"""        def baseline(x):
            os.environ['UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV']='0'
            return compensated_attention(x,k,v,**kwargs)
        def candidate(x):
            os.environ['UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV']='kv'
            return compensated_attention(x,k,v,**kwargs)
""")
source=source.replace('PREPACKED_COMPENSATED_ATTENTION','PREPACKED_ATTENTION_RUNTIME')
ast.parse(source);(root/'scripts/check_prepacked_attention_runtime.py').write_text(source)
print('PREPACKED_ATTENTION_RUNTIME_PATCHED original_jit_ast_and_lines=unchanged dynamic_kv_pack=every_replay small_shapes=original',root,flush=True)
