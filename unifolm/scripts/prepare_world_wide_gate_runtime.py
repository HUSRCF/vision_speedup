"""Install value-protected/gate-primary wide GEGLUs in a fresh source copy."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor) == 1 and 'UNIFOLM_WORLD_WIDE_GATE_PRIMARY' not in source
addition = '''    world_wide_gate_scope=os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY','0')
    if world_wide_gate_scope not in {'0','all','640','1280'}:
        raise ValueError('Invalid world wide gate scope')
    world_wide_gate_count=0
    if world_wide_gate_scope!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                prepacked_geglu_mode!='1'):
            raise ValueError('World wide gate requires accepted V2 policy')
        from .world_wide_gate_geglu import install_world_wide_gate_geglus
        world_wide_gate_count=install_world_wide_gate_geglus(model.model.diffusion_model,
            'gate',world_wide_gate_scope)
        if world_wide_gate_count!={'all':22,'640':10,'1280':12}[world_wide_gate_scope]:
            raise ValueError('Unexpected world wide gate module count')
    print(f'WORLD_WIDE_GATE_PRIMARY scope={world_wide_gate_scope} '
          f'modules={world_wide_gate_count} value_branch=accepted_3HalfBK64 '
          'gate_branch=primary master_bias_accum_gelu_output=fp32 '
          'decision=previous_installed_path weights=first_timed_warm',flush=True)
'''
source = source.replace(anchor, addition + anchor)
ast.parse(source)
path.write_text(source)
path = root / 'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor +
    '    echo "UNIFOLM_WORLD_WIDE_GATE_PRIMARY=${UNIFOLM_WORLD_WIDE_GATE_PRIMARY:-0}"\n')
path.write_text(source)
print('WORLD_WIDE_GATE_RUNTIME_PATCHED', root, flush=True)
