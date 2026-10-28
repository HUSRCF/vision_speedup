"""Make the previously validated grouped QKV installer scope explicit."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py'
source=path.read_text()
anchor="    primary_grouped_qkv_mode=os.environ.get('UNIFOLM_DECISION_PRIMARY_GROUPED_QKV','0')\n"
assert source.count(anchor)==1 and 'UNIFOLM_DECISION_PRIMARY_GROUPED_QKV_SCOPE' not in source
source=source.replace(anchor,anchor+"    primary_grouped_qkv_scope=os.environ.get('UNIFOLM_DECISION_PRIMARY_GROUPED_QKV_SCOPE','temporal')\n    if primary_grouped_qkv_scope not in {'temporal','all'}:\n        raise ValueError('Invalid primary grouped QKV scope')\n")
anchor='        primary_grouped_qkv_count=install_decision_primary_grouped_qkv(model.model.diffusion_model)\n'
assert source.count(anchor)==1
source=source.replace(anchor,'        primary_grouped_qkv_count=install_decision_primary_grouped_qkv(model.model.diffusion_model,scope=primary_grouped_qkv_scope)\n')
anchor='        if primary_grouped_qkv_count!=10:\n';assert source.count(anchor)==1
source=source.replace(anchor,"        if primary_grouped_qkv_count!=(10 if primary_grouped_qkv_scope=='temporal' else 15):\n")
anchor="scope=temporal world=previous master_accum_output=fp32 "
assert source.count(anchor)==1
source=source.replace(anchor,"scope={primary_grouped_qkv_scope} world=previous master_accum_output=fp32 ")
ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_DECISION_PRIMARY_GROUPED_QKV=${UNIFOLM_DECISION_PRIMARY_GROUPED_QKV:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'    echo "UNIFOLM_DECISION_PRIMARY_GROUPED_QKV_SCOPE=${UNIFOLM_DECISION_PRIMARY_GROUPED_QKV_SCOPE:-temporal}"\n')
path.write_text(source)
print('PRIMARY_GROUPED_QKV_SCOPE_RUNTIME_PATCHED',root,flush=True)
