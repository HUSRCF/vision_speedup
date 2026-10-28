"""Add one guarded dispatch without modifying attention's implementation."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/compensated_qkv.py'
source=path.read_text()
anchor='def fused_compensated_qkv(module, value):\n'
assert source.count(anchor)==1 and 'try_primary_grouped_qkv' not in source
addition='''    if getattr(module,'_decision_primary_grouped_qkv_installed',False):
        from .decision_primary_grouped_qkv import try_primary_grouped_qkv
        result=try_primary_grouped_qkv(module,value)
        if result is not None:return result
'''
source=source.replace(anchor,anchor+addition);ast.parse(source);path.write_text(source)
path=root/'src/unifolm_wma/utils/inference_runtime.py'
source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_DECISION_PRIMARY_GROUPED_QKV' not in source
addition='''    primary_grouped_qkv_mode=os.environ.get('UNIFOLM_DECISION_PRIMARY_GROUPED_QKV','0')
    if primary_grouped_qkv_mode not in {'0','1'}:
        raise ValueError('Primary grouped QKV mode must be0or1')
    primary_grouped_qkv_count=0
    if primary_grouped_qkv_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_DECISION_PRIMARY_NARROW','0') not in {'all','qkv'}):
            raise ValueError('Primary grouped QKV requires accepted primary C320 decision projections')
        from .decision_primary_grouped_qkv import install_decision_primary_grouped_qkv
        primary_grouped_qkv_count=install_decision_primary_grouped_qkv(model.model.diffusion_model)
        if primary_grouped_qkv_count!=10:
            raise ValueError(f'Unexpected primary grouped QKV count: {primary_grouped_qkv_count}')
    print(f'DECISION_PRIMARY_GROUPED_QKV requested={primary_grouped_qkv_mode} '
          f'modules={primary_grouped_qkv_count} scope=temporal world=previous master_accum_output=fp32 '
          'weights=first_timed_warm dynamic_input=each_timed_call',flush=True)
'''
source=source.replace(anchor,addition+anchor);ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'    echo "UNIFOLM_DECISION_PRIMARY_GROUPED_QKV=${UNIFOLM_DECISION_PRIMARY_GROUPED_QKV:-0}"\n');path.write_text(source)
print('PRIMARY_GROUPED_QKV_RUNTIME_PATCHED',root,flush=True)
