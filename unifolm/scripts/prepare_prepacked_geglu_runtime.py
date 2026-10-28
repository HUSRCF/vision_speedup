"""Enable wider fused GEGLU without changing the accepted GEMM precision."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
before = '    fused_qkv_mode = os.environ.get("UNIFOLM_FUSED_SELF_QKV", "0")\n'
after = '''    prepacked_geglu_mode = os.environ.get('UNIFOLM_PREPACKED_GEGLU', '0')
    if prepacked_geglu_mode not in {'0','1'}:
        raise ValueError('UNIFOLM_PREPACKED_GEGLU must be 0 or 1')
    prepacked_geglu_count = 0
    if prepacked_geglu_mode == '1':
        if (full_fp16_enabled or tf32x3_linears != 'triton' or
                compensated_linear_backend != 'scaled-fp16' or
                packed_half_weights != '1' or input_pack != 'wide' or
                half_linear_block_k != 64 or linear_group_m != 32):
            raise ValueError('Wide GEGLU requires accepted prepacked3Half BK64 group32 FP32 policy')
        from .triton_prepacked_geglu import install_prepacked_geglus
        prepacked_geglu_count = install_prepacked_geglus(model.model.diffusion_model)
    print(f'PREPACKED_GEGLU requested={prepacked_geglu_mode} '
          f'modules={prepacked_geglu_count} channels=640,1280 BK=64 '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'input_parts=refreshed_each_call',flush=True)
''' + before
assert source.count(before)==1
source = source.replace(before,after)
ast.parse(source)
path.write_text(source)
path = root/'scripts/team07_run.sh'
source = path.read_text()
before = '    echo "UNIFOLM_COMPENSATED_TEMPORAL_CONV=${UNIFOLM_COMPENSATED_TEMPORAL_CONV:-0}"\n'
assert source.count(before)==1
source = source.replace(before,before+'    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n')
path.write_text(source)
print('PREPACKED_GEGLU_RUNTIME_PATCHED',root,flush=True)
