"""Measure world initial512 shapes and verify the preserved decision wrapper."""
import json
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.initial_decision_linears import install_initial_decision_linears
from unifolm_wma.utils.initial_world_linears import install_initial_world_linears, make_forward
from unifolm_wma.utils.triton_half_compensated_linear import half_compensated_linear


@torch.no_grad()
def main():
    torch.manual_seed(571)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    for inputs, outputs in ((512, 512), (512, 4096), (2048, 512)):
        root = torch.nn.Module()
        root.init_attn = torch.nn.Sequential(torch.nn.Linear(inputs, outputs))
        root.outside = torch.nn.Linear(3, 3)
        assert install_initial_decision_linears(root) == 1
        root.eval().cuda()
        module = root.init_attn[0]
        original = module.forward
        keys = tuple(root.state_dict())
        assert install_initial_world_linears(root) == 1
        assert not getattr(root.outside, '_initial_world_linear_installed', False)
        value = torch.randn(40960, inputs, device='cuda') * .7
        torch.backends.cudnn.allow_tf32 = True
        torch.testing.assert_close(module(value), original(value), rtol=0, atol=0)
        torch.backends.cudnn.allow_tf32 = False
        actual = module(value)
        reference = half_compensated_linear(value, module.weight, module.bias,
            block_m=64, block_k=32, group_m=32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        sample = [0, 20480, 40959]
        exact = F.linear(value[sample].double(), module.weight.double(), module.bias.double())
        error = float((actual[sample].double() - exact).norm() / exact.norm())
        assert torch.isfinite(actual).all() and error < 1e-5
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
        module.weight.add_(.00001)
        module.bias.mul_(.95)
        eager = module(value)
        graph.replay()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        torch.backends.cudnn.allow_tf32 = True
        torch.testing.assert_close(module(value), original(value), rtol=0, atol=0)
        graph.replay()
        torch.testing.assert_close(captured, half_compensated_linear(value,
            module.weight, module.bias, block_m=64, block_k=32, group_m=32), rtol=0, atol=0)
        torch.backends.cudnn.allow_tf32 = False
        for fallback in ('training', 'grad', 'autocast', 'noncontiguous', 'shape'):
            module.train(fallback == 'training')
            x = (value.t().contiguous().t() if fallback == 'noncontiguous'
                 else value[:35] if fallback == 'shape' else value)
            with torch.set_grad_enabled(fallback == 'grad'):
                with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                    torch.testing.assert_close(module(x), F.linear(x, module.weight, module.bias),
                                               rtol=0, atol=0)
            module.eval()
        module.forward = original
        module.forward = make_forward(module)
        cold = torch.cuda.CUDAGraph()
        # The accepted compensated Linear handles a cold capture by splitting
        # weights inside GEMM, without allocating a static cache during capture.
        expected_cold = half_compensated_linear(value, module.weight, module.bias,
            block_m=64, block_k=32, group_m=32)
        with torch.cuda.graph(cold, stream=stream):
            cold_result = module(value)
        cold.replay()
        torch.testing.assert_close(cold_result, expected_cold, rtol=0, atol=0)
        assert keys == tuple(root.state_dict())
        assert all(p.dtype == torch.float32 for p in root.parameters())
        print('INITIAL_WORLD_LINEAR ' + json.dumps(dict(shape=[40960, inputs, outputs],
            fp64_relative=error, before_s=before, after_s=after, candidate_s=seconds,
            conservative_speedup=min(before, after) / seconds)), flush=True)
        del root, module, value, actual, reference, exact, captured, eager, graph
        del cold, cold_result, expected_cold, x, original
        torch.cuda.empty_cache()
    print('INITIAL_WORLD_LINEAR_CHECK passed=true actual_shapes=3 decision_previous_bitwise=true '
          'graph_input_weight_refresh=true cold_capture_uncached_weights=true native_fallbacks=5 '
          'master_weights=fp32 BK=32 full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
