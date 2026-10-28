"""Validate per-call input packing in the real Linear wrapper and graph cache."""
import json
from unittest.mock import patch
import torch
import torch.nn.functional as F
from unifolm_wma.utils.triton_compensated_linear import make_forward
from unifolm_wma.utils.triton_half_compensated_linear import half_compensated_linear


@torch.no_grad()
def main():
    torch.manual_seed(269)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    module = torch.nn.Linear(173, 1280).eval()
    keys = tuple(module.state_dict())
    def wrap(layer):
        layer.forward = make_forward(layer, min_matmul_macs=0,
            backend='scaled-fp16', cache_half_weights=True,
            group_m=32, block_k=64, pack_inputs=True)
    wrap(module)
    cpu = torch.randn(71, 173)
    torch.testing.assert_close(module(cpu), F.linear(cpu, module.weight, module.bias),
                               rtol=0, atol=0)
    module.cuda()
    value = cpu.cuda()
    def direct(layer, x):
        return half_compensated_linear(x, layer.weight, layer.bias,
                                       block_k=64, group_m=32)
    torch.testing.assert_close(module(value), direct(module, value), rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = module(value)
    module.weight.add_(.0001)
    module.bias.mul_(.9)
    value.mul_(.95)
    eager = module(value)
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    torch.testing.assert_close(eager, direct(module, value), rtol=0, atol=0)
    cold = torch.nn.Linear(173, 1280, device='cuda').eval()
    wrap(cold)
    expected = direct(cold, value)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cold_output = cold(value)
    graph.replay()
    torch.testing.assert_close(cold_output, expected, rtol=0, atol=0)
    for mode in ('training', 'grad', 'autocast', 'noncontiguous'):
        module.train(mode == 'training')
        x = value.t().contiguous().t() if mode == 'noncontiguous' else value
        with torch.set_grad_enabled(mode == 'grad'):
            with torch.autocast('cuda', enabled=mode == 'autocast', dtype=torch.float16):
                torch.testing.assert_close(module(x), F.linear(x, module.weight,
                                              module.bias), rtol=0, atol=0)
        module.eval()
    narrow = torch.nn.Linear(173, 70, device='cuda').eval()
    wrap(narrow)
    with patch('unifolm_wma.utils.triton_prepacked_input_linear.prepacked_input_linear',
               side_effect=AssertionError('Narrow output must bypass input packing')):
        torch.testing.assert_close(narrow(value), direct(narrow, value), rtol=0, atol=0)
    assert tuple(module.state_dict()) == keys
    assert all(p.dtype == torch.float32 for p in module.parameters())
    print('PREPACKED_INPUT_RUNTIME_CHECK '+json.dumps(dict(passed=True,
        bitwise=True, input_and_weight_refresh=True, cold_and_warm_capture=True,
        narrow_output_bypass=True, native_cpu_and_gpu_fallbacks=True,
        master_weights='fp32', full_video_quality='unverified')), flush=True)


if __name__ == '__main__':
    main()
