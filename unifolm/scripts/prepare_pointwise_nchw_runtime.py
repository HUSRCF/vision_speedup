"""Install only ten captured 1x1 NCHW shapes after other wrappers."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py'
source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_POINTWISE_NCHW' not in source
assert source.index('prepacked_geglu_count = install_prepacked_geglus(')<source.index(anchor)
addition='''    pointwise_nchw_mode=os.environ.get('UNIFOLM_POINTWISE_NCHW','0')
    if pointwise_nchw_mode not in {'0','1'}:
        raise ValueError('UNIFOLM_POINTWISE_NCHW must be0 or1')
    pointwise_nchw_count=0
    if pointwise_nchw_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Pointwise NCHW requires V2 FP32 decision-cudnn stage policy')
        from .pointwise_nchw_runtime import install_pointwise_nchw
        pointwise_nchw_count=install_pointwise_nchw(model.model.diffusion_model)
    print(f'POINTWISE_NCHW requested={pointwise_nchw_mode} modules={pointwise_nchw_count} '
          'scope=world_10_captured_shapes decision_head=native BK=64 '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'input_output_layout=direct_nchw',flush=True)
'''
source=source.replace(anchor,addition+anchor)
ast.parse(source)
path.write_text(source)
path=root/'scripts/team07_run.sh'
source=path.read_text()
anchor='    echo "UNIFOLM_DECISION_HALF_GEGLU=${UNIFOLM_DECISION_HALF_GEGLU:-0}"\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'    echo "UNIFOLM_POINTWISE_NCHW=${UNIFOLM_POINTWISE_NCHW:-0}"\n')
path.write_text(source)
print('POINTWISE_NCHW_RUNTIME_PATCHED',root,flush=True)
