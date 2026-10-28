"""Test BK64 cold/warm graph capture, FP64 error, cache invalidation and guards."""
import json
import torch
import torch.nn.functional as F
from unifolm_wma.utils.triton_compensated_linear import make_forward
from unifolm_wma.utils.triton_half_compensated_linear import half_compensated_linear


@torch.no_grad()
def main():
    torch.manual_seed(239)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    module = torch.nn.Linear(173, 70, device='cuda').eval()
    keys = tuple(module.state_dict())
    module.forward = make_forward(module, min_matmul_macs=0,
        backend='scaled-fp16', cache_half_weights=True, group_m=32, block_k=64)
    value = torch.randn(35, 173, device='cuda')
    def direct(x):
        return half_compensated_linear(x, module.weight, module.bias,
                                        group_m=32, block_k=64)
    # Load the uncached kernel outside capture; the wrapper's cache stays cold.
    expected = direct(value)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cold = module(value)
    graph.replay()
    torch.testing.assert_close(cold, expected, rtol=0, atol=0)
    module(value)
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
    torch.testing.assert_close(eager, direct(value), rtol=0, atol=0)
    exact = F.linear(value.double(), module.weight.double(), module.bias.double())
    error = float((eager.double()-exact).norm()/exact.norm())
    assert error < 1e-5, error
    for mode in ('training', 'grad', 'autocast', 'noncontiguous'):
        module.train(mode == 'training')
        x = value.t().contiguous().t() if mode == 'noncontiguous' else value
        with torch.set_grad_enabled(mode == 'grad'):
            with torch.autocast('cuda', enabled=mode == 'autocast', dtype=torch.float16):
                torch.testing.assert_close(module(x), F.linear(x, module.weight,
                                              module.bias), rtol=0, atol=0)
        module.eval()
    assert keys == tuple(module.state_dict())
    assert all(p.dtype == torch.float32 for p in module.parameters())
    print('HALF_LINEAR_REDUCTION_RUNTIME_CHECK '+json.dumps(dict(passed=True,
        block_k=64, group_m=32, cold_and_warm_capture=True,
        input_and_weight_refresh=True, fp64_relative_error=error,
        native_fallbacks=4, master_weights='fp32',
        full_video_quality='unverified')), flush=True)


if __name__ == '__main__':
    main()
