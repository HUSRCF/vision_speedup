"""Derive a new fusion kernel from the authoritative residual norm math."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1]);path=root/'src/unifolm_wma/utils/triton_residual_layer_norm.py'
source=path.read_text();tree=ast.parse(source)
original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_residual_layer_norm')
lines=source.splitlines(keepends=True)
kernel=''.join(lines[original.decorator_list[0].lineno-1:original.end_lineno])
kernel=kernel.replace('def _residual_layer_norm(', 'def _residual_norm_half_pack(')
anchor='Residual, Normalized,\n'
assert kernel.count(anchor)==1
kernel=kernel.replace(anchor,'Residual, Normalized, High, Low,\n')
anchor='BLOCK: tl.constexpr):'
assert kernel.count(anchor)==1
kernel=kernel.replace(anchor,'BLOCK: tl.constexpr, DEBUG: tl.constexpr):')
anchor='    tl.store(Normalized + offset, normalized, valid)'
assert kernel.count(anchor)==1
kernel=kernel.replace(anchor,'''    if DEBUG:
        tl.store(Normalized + offset, normalized, valid)
    high, low = _half_split(normalized, 1.)
    tl.store(High + offset, high, valid)
    tl.store(Low + offset, low, valid)''')
new=ast.parse(kernel).body[0]
# Every reduction, operation and residual store before the final normalized
# store remains exactly the previous AST. The original file is untouched.
assert [ast.dump(n,include_attributes=False) for n in original.body[:-1]]==[
    ast.dump(n,include_attributes=False) for n in new.body[:len(original.body)-1]]
output='import triton\nimport triton.language as tl\nfrom .triton_half_compensated_linear import _half_split\n\n\n'+kernel
ast.parse(output)
(root/'src/unifolm_wma/utils/triton_residual_norm_half_pack_kernel.py').write_text(output+'\n')
print('RESIDUAL_NORM_HALF_PACK_PREPARED original_math_AST=true old_source_unchanged=true',flush=True)
