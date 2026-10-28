"""Append a world-initial FFN installer without editing accepted kernels."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_INITIAL_BK32_NATIVE_GEGLU' not in source
addition='''    initial_native_mode=os.environ.get('UNIFOLM_INITIAL_BK32_NATIVE_GEGLU','0')
    if initial_native_mode not in {'0','1'}:
        raise ValueError('Invalid initial BK32 native GEGLU mode')
    initial_native_count=0
    if initial_native_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or _stage_tf32_mode()!='decision-cudnn'
                or stage_tf32_ddim_prefix_steps()!=0 or os.environ.get('UNIFOLM_INITIAL_WORLD_LINEARS')!='1'):
            raise ValueError('Initial BK32 native GEGLU requires accepted V2 stage policy')
        from .initial_bk32_native_geglu import install_initial_bk32_native_geglu
        initial_native_count=install_initial_bk32_native_geglu(model.model.diffusion_model)
        if initial_native_count==0:raise ValueError('No initial512 world FFN selected')
    print(f'INITIAL_BK32_NATIVE_GEGLU enabled={initial_native_mode} modules={initial_native_count} '
          'scope=world_initial512_M40960 original_BK32_three_products=true native_activation=true '
          'weight_pack=first_timed_warm version_refresh=same_storage input_conversion=each_call '
          'master_bias_accum_output=fp32 decision=previous',flush=True)
'''
updated=source.replace(anchor,addition+anchor);ast.parse(updated);path.write_text(updated)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n';assert source.count(anchor)==1
path.write_text(source.replace(anchor,anchor+'    echo "UNIFOLM_INITIAL_BK32_NATIVE_GEGLU=${UNIFOLM_INITIAL_BK32_NATIVE_GEGLU:-0}"\n'))
source=(root/'scripts/profile_initial_bk32_qkv_pair.py').read_text()
source=source.replace('UNIFOLM_INITIAL_BK32_QKV','UNIFOLM_INITIAL_BK32_NATIVE_GEGLU')
source=source.replace('INITIAL_BK32_QKV','INITIAL_BK32_NATIVE_GEGLU')
source=source.replace('initial_bk32_qkv','initial_bk32_native_geglu')
ast.parse(source);(root/'scripts/profile_initial_bk32_native_geglu_pair.py').write_text(source)
source=(root/'scripts/run_initial_bk32_qkv_pair.py').read_text()
source=source.replace('initial-bk32-qkv-graph-pair-','initial-bk32-native-geglu-graph-pair-')
source=source.replace("strftime('%Y%m%dT%H%M%SZ')","strftime('%Y%m%dT%H%M%S%fZ')")
source=source.replace('UNIFOLM_INITIAL_BK32_QKV','UNIFOLM_INITIAL_BK32_NATIVE_GEGLU')
source=source.replace('INITIAL_BK32_QKV','INITIAL_BK32_NATIVE_GEGLU')
source=source.replace('scripts/profile_initial_bk32_qkv_pair.py','scripts/profile_initial_bk32_native_geglu_pair.py')
anchor="flags['UNIFOLM_WORLD_ZERO_WEIGHT_LOW']='0'\n";assert source.count(anchor)==1
source=source.replace(anchor,'')
ast.parse(source);(root/'scripts/run_initial_bk32_native_geglu_pair.py').write_text(source)
print('INITIAL_BK32_NATIVE_GEGLU_PREPARED original_kernels_unchanged=true new_installer=true',root,flush=True)
