"""Change host scheduling after the old JIT functions, preserving their line numbers."""
import ast
from pathlib import Path
import sys

def transform(source):
    assert source.count('import torch\n')==1
    source=source.replace('import torch\n','import torch; import os\n')
    anchor='    bm = 64 if rows >= 2560 else 32\n'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'''    qkv_stages=os.environ.get('UNIFOLM_WORLD_ZERO_QKV_STAGES','2')
    if qkv_stages not in {'2','3-large'}:raise ValueError('Invalid zero-low QKV schedule')
    large_qkv={(40960,960,320),(10240,1920,640),(2560,3840,1280)}
    stages=(3 if qkv_stages=='3-large' and output_groups==3 and bias is None
        and (rows,channels,inner) in large_qkv
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cuda.matmul.allow_tf32 else 2)
''')
    anchor='        num_warps=4, num_stages=2, enable_fp_fusion=False)'
    assert source.count(anchor)==1
    source=source.replace(anchor,'        num_warps=4, num_stages=stages, enable_fp_fusion=False)')
    ast.parse(source);return source

if __name__=='__main__':
    path=Path(sys.argv[1])/'src/unifolm_wma/utils/triton_zero_weight_low_linear.py'
    path.write_text(transform(path.read_text()))
