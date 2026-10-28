"""Check composition with accepted F2medium and world-only5x8 guards."""
import json
from unittest.mock import patch
import torch
from benchmark_temporal_conv2d import graph_seconds
from benchmark_packed_spatial_conv import sampled_error
from unifolm_wma.utils.packed_spatial_runtime import make_forward
from unifolm_wma.utils.triton_winograd_f2 import make_f2_forward


@torch.no_grad()
def main():
    torch.manual_seed(587)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    for inputs in (1280, 2560):
        module = torch.nn.Conv2d(inputs, 1280, 3, padding=1).eval().cuda()
        value = torch.randn(16, inputs, 5, 8, device='cuda') * .7
        module.forward = make_f2_forward(module, scope='medium')
        original = module.forward
        original(value)  # Correct cuDNN first-call algorithm policy.
        keys = tuple(module.state_dict())
        forward = make_forward(module, 'all')
        result = forward(value)
        error = sampled_error(value, module.weight, module.bias, result)
        assert error < 3e-6 and torch.isfinite(result).all(), error
        before = graph_seconds(original, value, 10, 3)
        seconds = graph_seconds(forward, value, 10, 5)
        after = graph_seconds(original, value, 10, 3)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = forward(value)
        value.add_(.01)
        module.weight.add_(.00001)
        module.bias.mul_(.95)
        eager = forward(value)
        graph.replay()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        torch.backends.cudnn.allow_tf32 = True
        with patch('unifolm_wma.utils.packed_spatial_runtime.packed_spatial_conv',
                   side_effect=AssertionError('Decision entered packed spatial path')):
            torch.testing.assert_close(forward(value), original(value), rtol=0, atol=0)
        torch.backends.cudnn.allow_tf32 = False
        for fallback in ('training', 'grad', 'autocast', 'noncontiguous', 'shape', 'matmul_tf32'):
            module.train(fallback == 'training')
            torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
            x = (value.transpose(2, 3).contiguous().transpose(2, 3) if fallback == 'noncontiguous'
                 else torch.randn(16, inputs, 10, 16, device='cuda') if fallback == 'shape' else value)
            with torch.set_grad_enabled(fallback == 'grad'):
                with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                    with patch('unifolm_wma.utils.packed_spatial_runtime.packed_spatial_conv',
                               side_effect=AssertionError('Fallback entered packed spatial path')):
                        torch.testing.assert_close(forward(x), original(x), rtol=0, atol=0)
            module.eval()
            torch.backends.cuda.matmul.allow_tf32 = False
        if inputs == 2560:
            protected = make_forward(module, '1280')
            with patch('unifolm_wma.utils.packed_spatial_runtime.packed_spatial_conv',
                       side_effect=AssertionError('Protected concat entered packed spatial path')):
                torch.testing.assert_close(protected(value), original(value), rtol=0, atol=0)
        cold_forward = make_forward(module, 'all')
        expected = original(value)
        stream.wait_stream(torch.cuda.current_stream())
        cold = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold, stream=stream):
            cold_result = cold_forward(value)
        cold.replay()
        torch.testing.assert_close(cold_result, expected, rtol=0, atol=0)
        assert keys == tuple(module.state_dict())
        assert all(p.dtype == torch.float32 for p in module.parameters())
        print('PACKED_SPATIAL_RUNTIME ' + json.dumps(dict(inputs=inputs,
            fp64_relative=error, before_s=before, after_s=after, candidate_s=seconds,
            conservative_speedup=min(before, after) / seconds)), flush=True)
        assert min(before, after) > seconds
    print('PACKED_SPATIAL_RUNTIME_CHECK passed=true actual_shapes=2 decision_previous_bitwise=true '
          'F2medium_previous_bitwise=true native_fallbacks=6 cold_capture_fallback=true '
          'graph_input_weight_refresh=true protected_concat_excluded=true master_weights=fp32 '
          'full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
