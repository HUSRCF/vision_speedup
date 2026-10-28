"""Check world GEGLU including the existing decision wrapper and graph replay."""
import os
from unittest.mock import patch
import torch
from unifolm_wma.modules.attention import GEGLU
from unifolm_wma.utils.triton_compensated_linear import make_forward as projection_forward
from unifolm_wma.utils.triton_prepacked_geglu import install_prepacked_geglus
from unifolm_wma.utils.decision_primary_geglu import install_decision_primary_geglus
from unifolm_wma.utils.world_wide_gate_geglu import install_world_wide_gate_geglus, make_forward


@torch.no_grad()
def main():
    mode = 'gate'
    assert mode == 'gate'
    torch.manual_seed(547)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    for rows, channels in [(10240, 640), (2560, 1280), (640, 1280)]:
        module = GEGLU(channels, 4 * channels).eval().cuda()
        module.proj.forward = projection_forward(module.proj, backend='scaled-fp16',
            min_matmul_macs=0, cache_half_weights=True, group_m=32, block_k=64,
            pack_inputs=True)
        module.proj._tf32x3_installed = True
        assert install_prepacked_geglus(module) == 1
        assert install_decision_primary_geglus(module, 'both', 'all') == 1
        original = module.forward
        keys = tuple(module.state_dict())
        excluded = '1280' if channels == 640 else '640'
        assert install_world_wide_gate_geglus(module, mode, excluded) == 0
        assert module.forward is original
        assert install_world_wide_gate_geglus(module, mode) == 1
        value = torch.randn(rows, channels, device='cuda') * .7
        torch.backends.cudnn.allow_tf32 = True
        with patch('unifolm_wma.utils.world_wide_gate_geglu.branch_protected_geglu',
                   side_effect=AssertionError('Decision entered world GEGLU')):
            torch.testing.assert_close(module(value), original(value), rtol=0, atol=0)
        torch.backends.cudnn.allow_tf32 = False
        expected = module(value)
        assert expected.dtype == torch.float32 and torch.isfinite(expected).all()
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
        if rows == 640:
            for fallback in ('training', 'grad', 'autocast', 'noncontiguous',
                             'shape', 'matmul_tf32'):
                module.train(fallback == 'training')
                torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
                x = (value.t().contiguous().t() if fallback == 'noncontiguous'
                     else value[:35] if fallback == 'shape' else value)
                with torch.set_grad_enabled(fallback == 'grad'):
                    with torch.autocast('cuda', enabled=fallback == 'autocast',
                                        dtype=torch.float16):
                        with patch('unifolm_wma.utils.world_wide_gate_geglu.branch_protected_geglu',
                                   side_effect=AssertionError('Fallback entered world GEGLU')):
                            torch.testing.assert_close(module(x), original(x), rtol=0, atol=0)
                module.eval()
                torch.backends.cuda.matmul.allow_tf32 = False
            module.forward = original
            module.forward = make_forward(module, mode)
            cold_expected = original(value)
            cold = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cold, stream=stream):
                cold_result = module(value)
            cold.replay()
            torch.testing.assert_close(cold_result, cold_expected, rtol=0, atol=0)
            del cold, cold_result, cold_expected, x
        assert keys == tuple(module.state_dict())
        assert all(p.dtype == torch.float32 for p in module.parameters())
        print(f'WORLD_WIDE_GATE_GEGLU_RUNTIME mode={mode} shape={(rows, channels)} '
              'decision_previous_bitwise=true graph_input_weight_refresh=true', flush=True)
        del module, value, expected, eager, captured, graph, original
        torch.cuda.empty_cache()
    print(f'WORLD_WIDE_GATE_GEGLU_CHECK passed=true mode={mode} actual_shapes=3 '
          'decision_previous_bitwise=true scope_exclusion=true native_fallbacks=6 '
          'cold_capture_fallback=true graph_input_weight_refresh=true '
          'master_weights=fp32 full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
