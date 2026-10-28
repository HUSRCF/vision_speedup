"""Check C320 cached-weight world path with the accepted decision wrapper."""
import json
import os
from unittest.mock import patch
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.modules.attention import GEGLU
from unifolm_wma.utils.triton_compensated_geglu import compensated_geglu
from unifolm_wma.utils.decision_half_geglu import install_decision_half_geglus
from unifolm_wma.utils.world_prepacked_narrow_geglu import (
    install_world_prepacked_narrow_geglus, make_forward)


@torch.no_grad()
def main():
    tile = tuple(int(v) for v in os.environ['UNIFOLM_WORLD_NARROW_GEGLU_TILE'].split(','))
    torch.manual_seed(569)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    module = GEGLU(320, 1280).eval().cuda()
    module.forward = lambda x: compensated_geglu(x, module.proj.weight,
        module.proj.bias, group_m=16)
    module._compensated_geglu_installed = True
    assert install_decision_half_geglus(module) == 1
    original = module.forward
    keys = tuple(module.state_dict())
    assert install_world_prepacked_narrow_geglus(module, tile) == 1
    unrelated = GEGLU(320, 640).eval().cuda()
    assert install_world_prepacked_narrow_geglus(unrelated, tile) == 0
    value = torch.randn(40960, 320, device='cuda') * .7
    torch.backends.cudnn.allow_tf32 = True
    with patch('unifolm_wma.utils.world_prepacked_narrow_geglu.prepacked_geglu',
               side_effect=AssertionError('Decision entered world narrow GEGLU')):
        torch.testing.assert_close(module(value), original(value), rtol=0, atol=0)
    torch.backends.cudnn.allow_tf32 = False
    result = module(value)
    sample = [0, 20000, 40959]
    projected = F.linear(value[sample].double(), module.proj.weight.double(),
                         module.proj.bias.double())
    v, g = projected.chunk(2, dim=-1)
    exact = v * F.gelu(g)
    error = float((result[sample].double() - exact).norm() / exact.norm())
    assert torch.isfinite(result).all() and error < 2e-6, error
    before = graph_seconds(original, value, 10, 3)
    seconds = graph_seconds(module, value, 10, 5)
    after = graph_seconds(original, value, 10, 3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = module(value)
    value.add_(.01)
    module.proj.weight.add_(.00001)
    module.proj.bias.mul_(.95)
    eager = module(value)
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    torch.backends.cudnn.allow_tf32 = True
    torch.testing.assert_close(module(value), original(value), rtol=0, atol=0)
    torch.backends.cudnn.allow_tf32 = False
    # Exercise general PyTorch fallbacks against the native GEGLU, since the
    # accepted fused world baseline itself only supports inference inputs.
    native = GEGLU.forward.__get__(module)
    module.forward = native
    fallback_forward = make_forward(module, tile)
    for fallback in ('training', 'grad', 'autocast', 'noncontiguous', 'shape', 'matmul_tf32'):
        module.train(fallback == 'training')
        torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
        x = (value.t().contiguous().t() if fallback == 'noncontiguous'
             else value[:35] if fallback == 'shape' else value)
        with torch.set_grad_enabled(fallback == 'grad'):
            with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                with patch('unifolm_wma.utils.world_prepacked_narrow_geglu.prepacked_geglu',
                           side_effect=AssertionError('Fallback entered world narrow GEGLU')):
                    torch.testing.assert_close(fallback_forward(x), native(x), rtol=0, atol=0)
        module.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
    module.forward = original
    module.forward = make_forward(module, tile)
    expected_cold = original(value)
    cold = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold, stream=stream):
        cold_result = module(value)
    cold.replay()
    torch.testing.assert_close(cold_result, expected_cold, rtol=0, atol=0)
    assert keys == tuple(module.state_dict())
    assert all(p.dtype == torch.float32 for p in module.parameters())
    print('WORLD_PREPACKED_NARROW_GEGLU ' + json.dumps(dict(tile=tile,
        fp64_relative=error, before_s=before, after_s=after, candidate_s=seconds,
        conservative_speedup=min(before, after) / seconds)), flush=True)
    assert min(before, after) > seconds, 'Candidate must improve clean runtime timing'
    print('WORLD_PREPACKED_NARROW_GEGLU_CHECK passed=true actual_shape=true '
          'decision_previous_bitwise=true unrelated_scope_excluded=true native_fallbacks=6 '
          'cold_capture_fallback=true graph_input_weight_refresh=true '
          'master_weights=fp32 full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
