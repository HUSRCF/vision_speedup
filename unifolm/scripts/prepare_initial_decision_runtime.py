"""Surgically install decision-only BK32 initial linears in accepted V2."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
functions = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
assert {'_stage_tf32_mode', 'stage_tf32_ddim_prefix_steps'} <= functions
before = '    print(f"DENOISER_MODE requested={mode} case={dataset} wrapped={counts}", flush=True)\n'
after = '''    initial_decision_mode = os.environ.get('UNIFOLM_INITIAL_DECISION_LINEARS', '0')
    if initial_decision_mode not in {'0','1'}:
        raise ValueError('UNIFOLM_INITIAL_DECISION_LINEARS must be 0 or 1')
    initial_decision_count = 0
    if initial_decision_mode == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or
                stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('Initial decision linears require V2 FP32 and full decision-cudnn')
        from .initial_decision_linears import install_initial_decision_linears
        initial_decision_count = install_initial_decision_linears(model.model.diffusion_model)
    print(f'INITIAL_DECISION_LINEARS requested={initial_decision_mode} '
          f'modules={initial_decision_count} BK=32 world=native_fp32 '
          'master_output=fp32 weights=first_timed_warm', flush=True)
''' + before
assert source.count(before) == 1
source = source.replace(before, after)
ast.parse(source)
path.write_text(source)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
before = '    echo "UNIFOLM_COMPENSATED_TEMPORAL_CONV=${UNIFOLM_COMPENSATED_TEMPORAL_CONV:-0}"\n'
assert source.count(before) == 1
source = source.replace(before, before + '    echo "UNIFOLM_INITIAL_DECISION_LINEARS=${UNIFOLM_INITIAL_DECISION_LINEARS:-0}"\n')
path.write_text(source)
print('INITIAL_DECISION_RUNTIME_PATCHED', root, flush=True)
