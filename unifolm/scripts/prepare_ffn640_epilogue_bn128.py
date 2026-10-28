"""Select the exact 64x128 launch measured in the independent tile fixture."""
import ast
from pathlib import Path
import sys
path=Path(sys.argv[1])/'src/unifolm_wma/utils/ffn640_residual_epilogue.py'
source=path.read_text()
old='triton.cdiv(channels,64)';assert source.count(old)==1
source=source.replace(old,'triton.cdiv(channels,128)')
old='rows,channels,inner,bias is not None,True,bm,64,64,32,';assert source.count(old)==1
source=source.replace(old,'rows,channels,inner,bias is not None,True,bm,128,64,32,')
ast.parse(source);path.write_text(source)
print('FFN640_EPILOGUE_BN128_PREPARED original_BK64_two_product_math=true BM64_BN128_group32_warps4_stages2=true',flush=True)
