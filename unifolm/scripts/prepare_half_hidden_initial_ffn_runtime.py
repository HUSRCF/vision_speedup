"""Opt into direct Half storage of the final initial FFN GEMM input."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py'
source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_HALF_HIDDEN_INITIAL_FFN' not in source
addition='''    half_hidden_initial_mode=os.environ.get('UNIFOLM_HALF_HIDDEN_INITIAL_FFN','0')
    if half_hidden_initial_mode not in {'0','1'}:
        raise ValueError('Half hidden initial FFN mode must be0or1')
    half_hidden_initial_count=0
    if half_hidden_initial_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or initial_primary_scope!='ffn' or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_INITIAL_PRIMARY_GEGLU','0')!='0' or
                os.environ.get('UNIFOLM_NVRTC_GEGLU','0')!='fmad'):
            raise ValueError('Half hidden FFN requires separate accepted primaryFFN/native policy')
        from .half_hidden_initial_ffn import install_half_hidden_initial_ffn
        half_hidden_initial_count=install_half_hidden_initial_ffn(model.model.diffusion_model)
        if half_hidden_initial_count!=1:
            raise ValueError('Unexpected initial FFN count')
    print(f'HALF_HIDDEN_INITIAL_FFN requested={half_hidden_initial_mode} '
          f'modules={half_hidden_initial_count} world=previous '
          'hidden=existing_final_gemm_half_input master_bias_accum_output=fp32 '
          'weights=first_timed_warm dynamic_input=each_timed_call',flush=True)
'''
source=source.replace(anchor,addition+anchor)
ast.parse(source)
path.write_text(source)
path=root/'scripts/team07_run.sh'
source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+
    '    echo "UNIFOLM_HALF_HIDDEN_INITIAL_FFN=${UNIFOLM_HALF_HIDDEN_INITIAL_FFN:-0}"\n')
path.write_text(source)
print('HALF_HIDDEN_INITIAL_FFN_RUNTIME_PATCHED',root,flush=True)
