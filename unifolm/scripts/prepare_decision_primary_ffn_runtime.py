"""Install a named/stage-scoped FFN arithmetic experiment on accepted runtime."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py'
source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_DECISION_PRIMARY_FFN' not in source
addition='''    primary_ffn_mode=os.environ.get('UNIFOLM_DECISION_PRIMARY_FFN','0')
    if primary_ffn_mode not in {'0','all','640','1280'}:
        raise ValueError('UNIFOLM_DECISION_PRIMARY_FFN must be0,all,640,or1280')
    primary_ffn_count=0
    if primary_ffn_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                vendor_ffn_mode!='chunked'):
            raise ValueError('Primary FFN requires accepted V2 decision chunked FFN policy')
        from .decision_primary_ffn import install_decision_primary_ffns
        primary_ffn_count=install_decision_primary_ffns(model.model.diffusion_model,primary_ffn_mode)
    print(f'DECISION_PRIMARY_FFN requested={primary_ffn_mode} modules={primary_ffn_count} '
          'scope=decision_ffn_output world=accepted_compensation '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'input_conversion=each_timed_call',flush=True)
'''
source=source.replace(anchor,addition+anchor)
ast.parse(source)
path.write_text(source)
path=root/'scripts/team07_run.sh'
source=path.read_text()
anchor='    echo "UNIFOLM_DECISION_VENDOR_FFN=${UNIFOLM_DECISION_VENDOR_FFN:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'    echo "UNIFOLM_DECISION_PRIMARY_FFN=${UNIFOLM_DECISION_PRIMARY_FFN:-0}"\n')
path.write_text(source)
print('DECISION_PRIMARY_FFN_RUNTIME_PATCHED',root,flush=True)
