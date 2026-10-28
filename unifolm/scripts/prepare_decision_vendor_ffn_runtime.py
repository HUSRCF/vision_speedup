"""Add only a decision FFN vendor-backend dispatcher to accepted V2."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
functions = {node.name for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
assert {'_stage_tf32_mode','stage_tf32_ddim_prefix_steps'}<=functions
before = '    print(f"DENOISER_MODE requested={mode} case={dataset} wrapped={counts}", flush=True)\n'
after = '''    vendor_ffn_mode = os.environ.get('UNIFOLM_DECISION_VENDOR_FFN','0')
    if vendor_ffn_mode not in {'0','full','chunked'}:
        raise ValueError('UNIFOLM_DECISION_VENDOR_FFN must be 0,full,or chunked')
    vendor_ffn_count = 0
    if vendor_ffn_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                compensated_linear_backend!='scaled-fp16' or packed_half_weights!='1'):
            raise ValueError('Vendor FFN requires accepted V2 FP32 and decision-cudnn policy')
        from .decision_vendor_ffn import install_decision_vendor_ffns
        vendor_ffn_count = install_decision_vendor_ffns(model.model.diffusion_model,vendor_ffn_mode)
    print(f'DECISION_VENDOR_FFN requested={vendor_ffn_mode} modules={vendor_ffn_count} '
          'scope=decision_ffn_output640,1280 world=accepted_shortBK64 '
          'master_bias_accum_output=fp32 weights=first_timed_warm',flush=True)
''' + before
assert source.count(before)==1
source = source.replace(before,after)
ast.parse(source)
path.write_text(source)
path = root/'scripts/team07_run.sh'
source = path.read_text()
before = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(before)==1
source = source.replace(before,before+'    echo "UNIFOLM_DECISION_VENDOR_FFN=${UNIFOLM_DECISION_VENDOR_FFN:-0}"\n')
path.write_text(source)
print('DECISION_VENDOR_FFN_RUNTIME_PATCHED',root,flush=True)
