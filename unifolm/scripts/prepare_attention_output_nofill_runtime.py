"""Patch only public output allocation; preserve JIT ASTs and line numbers."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])


def kernels(source):
    return {n.name: (ast.dump(n, include_attributes=False), n.lineno)
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.decorator_list}


for filename, function, kind in [
        ('triton_compensated_attention.py', 'compensated_attention', 'spatial'),
        ('triton_temporal_attention_fp32.py', 'fused_temporal_attention_fp32', 'temporal')]:
    path = root/'src/unifolm_wma/utils'/filename
    source = path.read_text()
    before = kernels(source)
    functions = [n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name == function]
    assert len(functions) == 1
    assignments = [n for n in ast.walk(functions[0])
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'output'
                   and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'torch.empty_like']
    assert len(assignments) == 1
    call = assignments[0].value
    assert ast.unparse(call) == 'torch.empty_like(query)' and call.lineno == call.end_lineno
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:call.lineno-1])) + call.col_offset
    end = sum(map(len, lines[:call.end_lineno-1])) + call.end_col_offset
    source = source[:start] + f"empty_attention_output('{kind}', query)" + source[end:]
    assert source.count('import torch\n') == 1
    source = source.replace('import torch\n',
        'import torch; from .attention_buffer_allocation import empty_like as empty_attention_output\n', 1)
    assert kernels(source) == before
    path.write_text(source)

path = root/'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor +
    '    echo "UNIFOLM_ATTENTION_OUTPUT_NOFILL=${UNIFOLM_ATTENTION_OUTPUT_NOFILL:-0}"\n')
path.write_text(source)

# Reuse the existing actual-input, alternating-order two-stage graph diagnostic.
source = (root/'scripts/profile_initial_bk32_qkv_pair.py').read_text()
assert source.count("'UNIFOLM_INITIAL_BK32_QKV'") == 4
source = source.replace("'UNIFOLM_INITIAL_BK32_QKV'", "'UNIFOLM_ATTENTION_OUTPUT_NOFILL'")
source = source.replace("[('baseline','0'),('candidate','1')]", "[('baseline','0'),('candidate','all')]")
block = """                    if mode != '0':
                        from unifolm_wma.utils.initial_bk32_qkv import install_initial_bk32_qkv
                        assert install_initial_bk32_qkv(model.model.diffusion_model) > 0
"""
assert source.count(block) == 1
source = source.replace(block, '').replace('INITIAL_BK32_QKV', 'ATTENTION_OUTPUT_NOFILL')
ast.parse(source)
(root/'scripts/profile_attention_output_nofill_pair.py').write_text(source)
source = (root/'scripts/run_initial_bk32_qkv_pair.py').read_text()
source = source.replace("'initial-bk32-qkv-graph-pair-'", "'attention-output-nofill-graph-pair-'")
source = source.replace("flags['UNIFOLM_INITIAL_BK32_QKV']='0'",
                        "assert flags['UNIFOLM_INITIAL_BK32_QKV']=='1'\nflags['UNIFOLM_ATTENTION_OUTPUT_NOFILL']='0'")
source = source.replace('scripts/profile_initial_bk32_qkv_pair.py', 'scripts/profile_attention_output_nofill_pair.py')
source = source.replace('INITIAL_BK32_QKV_', 'ATTENTION_OUTPUT_NOFILL_')
ast.parse(source)
(root/'scripts/run_attention_output_nofill_pair.py').write_text(source)
print('ATTENTION_OUTPUT_NOFILL_RUNTIME_PATCHED helpers=2 jit_asts_and_lines=unchanged', root, flush=True)
