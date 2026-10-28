"""Validate actual wrapper chain, decision isolation and mutable graph inputs."""
import json
from unittest.mock import patch
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.modules.attention import GEGLU
from unifolm_wma.utils.triton_compensated_geglu import install_compensated_geglus
from unifolm_wma.utils.decision_half_geglu import install_decision_half_geglus
from unifolm_wma.utils.world_prepacked_narrow_geglu import install_world_prepacked_narrow_geglus
from unifolm_wma.utils.decision_c320_primary_geglu import install_decision_c320_primary_geglus
from unifolm_wma.utils.world_bk64_c320_geglu import install_world_c320_primary_geglus, make_forward


@torch.no_grad()
def main():
    torch.manual_seed(661)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    value = torch.randn(40960, 320, device='cuda') * .7
    for mode in ('value', 'gate'):
        bn = 32 if mode == 'value' else 64
        module = GEGLU(320, 1280).eval().cuda()
        assert install_compensated_geglus(module) == 1
        assert install_decision_half_geglus(module) == 1
        assert install_world_prepacked_narrow_geglus(module, (64, 32, 16)) == 1
        assert install_decision_c320_primary_geglus(module, 'both', 64) == 1
        original, keys = module.forward, tuple(module.state_dict())
        assert install_world_c320_primary_geglus(module, mode, bn) == 1
        assert install_world_c320_primary_geglus(GEGLU(640, 2560).eval().cuda(), mode, bn) == 0
        torch.backends.cudnn.allow_tf32 = True
        with patch('unifolm_wma.utils.world_bk64_c320_geglu.primary_narrow_geglu',
                   side_effect=AssertionError('Decision entered world experiment')):
            torch.testing.assert_close(module(value), original(value), rtol=0, atol=0)
        torch.backends.cudnn.allow_tf32 = False
        result = module(value)
        sample = [0, 20000, 40959]
        v, g = F.linear(value[sample].double(), module.proj.weight.double(),
                        module.proj.bias.double()).chunk(2, dim=-1)
        exact = v * F.gelu(g)
        error = float((result[sample].double() - exact).norm() / exact.norm())
        assert error < 1e-3 and torch.isfinite(result).all(), error
        before = graph_seconds(original, value, 5, 3)
        seconds = graph_seconds(module, value, 5, 3)
        after = graph_seconds(original, value, 5, 3)
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
        for fallback in ('training', 'grad', 'autocast', 'noncontiguous', 'shape', 'matmul_tf32'):
            module.train(fallback == 'training')
            torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
            x = (value.t().contiguous().t() if fallback == 'noncontiguous'
                 else value[:35] if fallback == 'shape' else value)
            with torch.set_grad_enabled(fallback == 'grad'):
                with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                    with patch('unifolm_wma.utils.world_bk64_c320_geglu.primary_narrow_geglu',
                               side_effect=AssertionError('Fallback entered world experiment')):
                        torch.testing.assert_close(module(x), original(x), rtol=0, atol=0)
            module.eval()
            torch.backends.cuda.matmul.allow_tf32 = False
        module.forward = original
        module.forward = make_forward(module, mode, bn)
        expected = original(value)
        stream.wait_stream(torch.cuda.current_stream())
        cold = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold, stream=stream):
            cold_result = module(value)
        cold.replay()
        torch.testing.assert_close(cold_result, expected, rtol=0, atol=0)
        assert keys == tuple(module.state_dict())
        assert all(p.dtype == torch.float32 for p in module.parameters())
        print('WORLD_BK64_C320_GEGLU_GUARD ' + json.dumps(dict(mode=mode,
            fp64_relative=error, before_s=before, after_s=after, candidate_s=seconds,
            conservative_speedup=min(before, after)/seconds)), flush=True)
    print('WORLD_BK64_C320_GEGLU_CHECK passed=true decision_previous_bitwise=true '
          'actual_shape=true unrelated_scope_excluded=true native_fallbacks=6 '
          'cold_capture_fallback=true graph_input_weight_refresh=true '
          'master_weights=fp32 full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
