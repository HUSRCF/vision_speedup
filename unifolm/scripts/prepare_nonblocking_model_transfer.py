"""Keep all model transfer work inside its original timed stage."""
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root/'scripts/evaluation/world_model_interaction.py'
source = path.read_text()
anchor = '        model = model.cuda()\n'
assert source.count(anchor) == 1 and 'UNIFOLM_MODEL_TO_GPU_NONBLOCKING' not in source
source = source.replace(anchor, '''        transfer_mode = os.environ.get('UNIFOLM_MODEL_TO_GPU_NONBLOCKING', '0')
        if transfer_mode not in {'0', '1'}:
            raise ValueError('Invalid model GPU transfer mode')
        if transfer_mode == '1':
            model = model.to(device='cuda', non_blocking=True)
        else:
            model = model.cuda()
        print(f'MODEL_TO_GPU_NONBLOCKING enabled={transfer_mode} '
              'scope=original_timed_model_to_gpu dtype=preserved', flush=True)
''')
assert source.count('model = model.to(') >= 1
ast.parse(source)
# Retain the original explicit synchronization before optimization setup.
start = source.index('with (timing.stage("model_to_gpu")')
end = source.index('with (timing.stage("inference_optimization_setup")', start)
assert 'torch.cuda.synchronize()' in source[start:end]
path.write_text(source)
path = root/'scripts/team07_run.sh'
source = path.read_text()
anchor = '    echo "UNIFOLM_PREPACKED_GEGLU=${UNIFOLM_PREPACKED_GEGLU:-0}"\n'
assert source.count(anchor) == 1
path.write_text(source.replace(anchor, anchor +
    '    echo "UNIFOLM_MODEL_TO_GPU_NONBLOCKING=${UNIFOLM_MODEL_TO_GPU_NONBLOCKING:-0}"\n'))
print('NONBLOCKING_MODEL_TRANSFER_PATCHED timed_scope_and_sync=preserved', root, flush=True)
