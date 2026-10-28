"""Install a stage-restricted C320 experiment, preserving world arithmetic."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/triton_half_compensated_geglu.py'
source = path.read_text()
assert 'GROUP_M' not in source
before = '''                            BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
'''
after = '''                            BN: tl.constexpr, GROUP_M: tl.constexpr):
    if GROUP_M:
        pid = tl.program_id(0)
        row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(C, BN)
        group = pid // (GROUP_M * col_tiles)
        first_row = group * GROUP_M
        group_rows = tl.minimum(GROUP_M, row_tiles - first_row)
        local = pid % (GROUP_M * col_tiles)
        row_tile = first_row + local % group_rows
        col_tile = local // group_rows
    else:
        row_tile, col_tile = tl.program_id(0), tl.program_id(1)
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)
'''
assert source.count(before) == 1
source = source.replace(before, after)
before = 'def half_compensated_geglu(value, weight, bias=None, *, block_m=64, block_n=32):\n'
after = '''def half_compensated_geglu(value, weight, bias=None, *, block_m=64, block_n=32, group_m=0):
    if group_m not in {0, 4, 8, 16, 32}:
        raise ValueError('Unsupported Half GEGLU row grouping')
'''
assert source.count(before) == 1
source = source.replace(before, after)
before = '    _half_compensated_geglu[(triton.cdiv(rows, block_m), triton.cdiv(channels, block_n))](\n'
after = '''    row_tiles, col_tiles = triton.cdiv(rows, block_m), triton.cdiv(channels, block_n)
    grid = (row_tiles*col_tiles,) if group_m else (row_tiles, col_tiles)
    _half_compensated_geglu[grid](
'''
assert source.count(before) == 1
source = source.replace(before, after)
before = '        rows, channels, reduction, bias is not None, block_m, block_n,\n'
assert source.count(before) == 1
source = source.replace(before, before.rstrip('\n')+' group_m,\n')
ast.parse(source)
path.write_text(source)
path = root/'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    fused_qkv_mode = os.environ.get("UNIFOLM_FUSED_SELF_QKV", "0")\n'
addition = '''    decision_half_geglu_mode = os.environ.get('UNIFOLM_DECISION_HALF_GEGLU', '0')
    if decision_half_geglu_mode not in {'0', '1'}:
        raise ValueError('UNIFOLM_DECISION_HALF_GEGLU must be 0 or 1')
    decision_half_geglu_count = 0
    if decision_half_geglu_mode == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('Decision Half GEGLU requires V2 FP32 decision-cudnn stage policy')
        from .decision_half_geglu import install_decision_half_geglus
        decision_half_geglu_count = install_decision_half_geglus(model.model.diffusion_model)
    print(f'DECISION_HALF_GEGLU requested={decision_half_geglu_mode} '
          f'modules={decision_half_geglu_count} channels=320 BK=32 group=16 '
          'world=accepted_3TF32 master_bias_accum_output=fp32', flush=True)
'''
assert source.count(anchor) == 1 and 'UNIFOLM_DECISION_HALF_GEGLU' not in source
assert source.index('compensated_geglu_count = install_compensated_geglus(') < source.index(anchor)
assert source.index('prepacked_geglu_count = install_prepacked_geglus(') < source.index(anchor)
source = source.replace(anchor, addition+anchor)
ast.parse(source)
path.write_text(source)
path = root/'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor+'    echo "UNIFOLM_DECISION_HALF_GEGLU=${UNIFOLM_DECISION_HALF_GEGLU:-0}"\n')
path.write_text(source)
print('DECISION_HALF_GEGLU_RUNTIME_PATCHED', root, flush=True)
