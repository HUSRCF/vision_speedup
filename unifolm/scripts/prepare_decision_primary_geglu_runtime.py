"""Add only decision wide-GEGLU dispatch, after accepted wide wrappers exist."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py'
source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_DECISION_PRIMARY_GEGLU' not in source
assert source.index('prepacked_geglu_count = install_prepacked_geglus(')<source.index(anchor)
addition='''    primary_geglu_mode=os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU','0')
    primary_geglu_scope=os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU_SCOPE','all')
    if primary_geglu_mode not in {'0','value','both'} or primary_geglu_scope not in {'all','640','1280'}:
        raise ValueError('Invalid decision primary GEGLU mode or scope')
    primary_geglu_count=0
    if primary_geglu_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                prepacked_geglu_mode!='1'):
            raise ValueError('Primary GEGLU requires accepted V2 decision-cudnn and wideGEGLU')
        from .decision_primary_geglu import install_decision_primary_geglus
        primary_geglu_count=install_decision_primary_geglus(model.model.diffusion_model,
            primary_geglu_mode,primary_geglu_scope)
    print(f'DECISION_PRIMARY_GEGLU requested={primary_geglu_mode} '
          f'channel_scope={primary_geglu_scope} modules={primary_geglu_count} '
          'world=accepted3Half master_bias_gelu_output=fp32 '
          'weights=first_timed_warm inputs=refreshed_each_timed_call',flush=True)
'''
source=source.replace(anchor,addition+anchor)
ast.parse(source)
path.write_text(source)
path=root/'scripts/team07_run.sh'
source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+
    '    echo "UNIFOLM_DECISION_PRIMARY_GEGLU=${UNIFOLM_DECISION_PRIMARY_GEGLU:-0}"\n'+
    '    echo "UNIFOLM_DECISION_PRIMARY_GEGLU_SCOPE=${UNIFOLM_DECISION_PRIMARY_GEGLU_SCOPE:-all}"\n')
path.write_text(source)
print('DECISION_PRIMARY_GEGLU_RUNTIME_PATCHED',root,flush=True)
