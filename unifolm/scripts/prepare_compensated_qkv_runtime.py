"""Patch only QKV dispatch and opt-in configuration in a coherent snapshot."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/unifolm_wma/modules/attention.py'
source = path.read_text()
before = '''        if spatial_self_attn and self._fused_self_qkv_enabled:
            q, k, v = self._project_self_qkv(x).chunk(3, dim=-1)
'''
after = '''        if spatial_self_attn and (self._fused_self_qkv_enabled or
                getattr(self, '_compensated_self_qkv_enabled', False)):
            if getattr(self, '_compensated_self_qkv_enabled', False):
                from unifolm_wma.utils.compensated_qkv import fused_compensated_qkv
                q, k, v = fused_compensated_qkv(self, x)
            else:
                q, k, v = self._project_self_qkv(x).chunk(3, dim=-1)
'''
assert source.count(before) == 2
source = source.replace(before, after)
before = '        elif not (spatial_self_attn and self._fused_self_qkv_enabled):'
after = '''        elif not (spatial_self_attn and (self._fused_self_qkv_enabled or
                getattr(self, '_compensated_self_qkv_enabled', False))):'''
assert source.count(before) == 2
source = source.replace(before, after)
ast.parse(source)
path.write_text(source)

path = root / 'src/unifolm_wma/utils/inference_runtime.py'
source = path.read_text()
before = '    cross_attention_mode = os.environ.get(\n'
after = '''    compensated_qkv = os.environ.get('UNIFOLM_COMPENSATED_SELF_QKV', '0')
    if compensated_qkv not in {'0', '1'}:
        raise ValueError('UNIFOLM_COMPENSATED_SELF_QKV must be 0 or 1')
    compensated_qkv_count = 0
    if compensated_qkv == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or fused_qkv_enabled or
                compensated_linear_backend != 'scaled-fp16' or
                os.environ.get('UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK', '0') != 'wide' or
                os.environ.get('UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE', '0') != '1' or
                os.environ.get('UNIFOLM_COMPENSATED_LINEAR_GROUP_M', '0') != '32' or
                os.environ.get('UNIFOLM_COMPENSATED_LINEAR_BLOCK_K', '32') != '64'):
            raise ValueError('Compensated QKV requires V2 FP32 prepacked BK64/group32 base')
        from unifolm_wma.utils.compensated_qkv import install_compensated_qkv
        compensated_qkv_count = install_compensated_qkv(model.model.diffusion_model)
    print(f'COMPENSATED_SELF_QKV requested={compensated_qkv} '
          f'modules={compensated_qkv_count} output_layout=three_contiguous_fp32 '
          'input_conversion_in_timing=true', flush=True)
''' + before
assert source.count(before) == 1
source = source.replace(before, after)
ast.parse(source)
path.write_text(source)

path = root / 'scripts/team07_run.sh'
lines = path.read_text().splitlines(keepends=True)
for i, line in enumerate(lines):
    if 'UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK=' in line and 'echo ' in line:
        lines.insert(i+1, '    echo "UNIFOLM_COMPENSATED_SELF_QKV=${UNIFOLM_COMPENSATED_SELF_QKV:-0}"\n')
        break
else:
    raise RuntimeError('Missing provenance anchor')
path.write_text(''.join(lines))
print('COMPENSATED_QKV_RUNTIME_PATCHED', root, flush=True)
