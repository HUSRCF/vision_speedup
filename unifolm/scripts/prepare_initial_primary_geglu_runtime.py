"""Add bitwise native-activation fusion to an accepted primary FFN source copy."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
anchor = '    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_INITIAL_PRIMARY_GEGLU' not in source
addition = '''    initial_primary_geglu_mode=os.environ.get('UNIFOLM_INITIAL_PRIMARY_GEGLU','0')
    if initial_primary_geglu_mode not in {'0','1'}:
        raise ValueError('Initial primary GEGLU must be0or1')
    initial_primary_geglu_count=0
    if initial_primary_geglu_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or initial_primary_scope!='ffn' or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_NVRTC_GEGLU','0')!='fmad'):
            raise ValueError('Initial primary fusion requires accepted FFN2 policy/native activation')
        from .initial_primary_geglu import install_initial_primary_geglu
        initial_primary_geglu_count=install_initial_primary_geglu(model.model.diffusion_model)
        if initial_primary_geglu_count!=1:
            raise ValueError('Unexpected initial512 primary GEGLU count')
    print(f'INITIAL_PRIMARY_GEGLU requested={initial_primary_geglu_mode} '
          f'modules={initial_primary_geglu_count} stage=decision tile=64,128,32,4,2 '
          'world=previous activation=native_sm80 master_bias_accum_output=fp32 '
          'weights=first_timed_warm input_cast=each_timed_call',flush=True)
'''
source = source.replace(anchor,addition+anchor)
ast.parse(source)
path.write_text(source)
path = root/'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
source = source.replace(anchor,anchor+
    '    echo "UNIFOLM_INITIAL_PRIMARY_GEGLU=${UNIFOLM_INITIAL_PRIMARY_GEGLU:-0}"\n')
path.write_text(source)
print('INITIAL_PRIMARY_GEGLU_RUNTIME_PATCHED',root,flush=True)
