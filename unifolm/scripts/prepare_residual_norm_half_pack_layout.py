"""Keep Half stores from selecting a wider layout for the norm reduction."""
import ast
from pathlib import Path
import runpy
import sys
root=Path(sys.argv[1])
runpy.run_path(str(root/'scripts/prepare_residual_norm_half_pack.py'),run_name='__main__')
p=root/'src/unifolm_wma/utils/triton_residual_norm_half_pack_kernel.py';source=p.read_text()
anchor='    high, low = _half_split(normalized, 1.)\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'''    # Hide only Half-store address vectorization from AxisInfo. FP32 loads
    # and the residual store still determine the original four-element
    # reduction layout. This identity leaves every address unchanged.
    pack_offset = tl.inline_asm_elementwise("mov.b32 $0, $1;", constraints="=r,r",
        args=[offset], dtype=tl.int32, is_pure=True, pack=1)
''')
assert source.count('High + offset')==source.count('Low + offset')==1
source=source.replace('High + offset','High + pack_offset').replace('Low + offset','Low + pack_offset')
ast.parse(source);p.write_text(source)
print('RESIDUAL_NORM_HALF_PACK_LAYOUT_PREPARED opaque_half_store_address_only=true '
      'all_norm_math_and_FP32_addresses_unchanged=true',flush=True)
