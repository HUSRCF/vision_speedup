"""Exercise the installed wrapper, graph replay and retained native stages."""
import os
from unittest.mock import patch
from check_compensated_temporal_runtime import main as check_accepted
from unifolm_wma.utils import triton_implicit_temporal_conv as implicit

assert os.environ.get('UNIFOLM_IMPLICIT_TEMPORAL_CONV') == '1'
original = implicit.implicit_temporal_conv
shapes = set()


def observed(value, *args, **kwargs):
    shapes.add(tuple(value.shape))
    return original(value, *args, **kwargs)


with patch.object(implicit, 'implicit_temporal_conv', side_effect=observed):
    check_accepted()
assert shapes == {(1,320,16,40,64), (1,640,16,20,32)}, shapes
print('IMPLICIT_TEMPORAL_RUNTIME_CHECK passed=true actual_fused_shapes=2 '
      'unfused_shapes=2 graph_input_weight_refresh=true tf32_native_bitwise=true '
      'reference=accepted_compensated_temporal bitwise=true full_video_quality=unverified',
      flush=True)
