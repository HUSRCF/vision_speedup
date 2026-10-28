"""Exercise masked tiles, graph refresh, original fallbacks, and FP32 weights."""
import copy
import json
import torch
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_compensated_linear import (
    make_forward, install_compensated_linears, compensated_linear)
from unifolm_wma.utils.triton_half_compensated_linear import half_compensated_linear


@torch.no_grad()
def main():
    torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for device in ('cpu', 'cuda'):
        reference = torch.nn.Linear(37, 70, device=device).eval()
        candidate = copy.deepcopy(reference)
        keys = tuple(candidate.state_dict())
        weights = {k: v.clone() for k, v in candidate.state_dict().items()}
        candidate.forward = make_forward(candidate, min_matmul_macs=0, backend='scaled-fp16')
        value = torch.randn(35, 37, device=device)
        if device == 'cpu':
            torch.testing.assert_close(candidate(value), reference(value), rtol=0, atol=0)
        else:
            actual = candidate(value)
            expected = half_compensated_linear(value, candidate.weight, candidate.bias)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            precise = torch.nn.functional.linear(value.double(), candidate.weight.double(), candidate.bias.double())
            assert float((actual.double()-precise).norm()/precise.norm()) < 3e-6
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): candidate(value)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream): output = candidate(value)
            value.mul_(0.95)
            graph.replay()
            torch.testing.assert_close(output, candidate(value), rtol=0, atol=0)
            with torch.autocast('cuda', dtype=torch.float16):
                torch.testing.assert_close(candidate(value), reference(value), rtol=0, atol=0)
        noncontiguous = torch.randn(37, 35, device=device).t()
        torch.testing.assert_close(candidate(noncontiguous), reference(noncontiguous), rtol=0, atol=0)
        candidate.train(); reference.train()
        torch.testing.assert_close(candidate(value), reference(value), rtol=0, atol=0)
        candidate.eval(); reference.eval()
        with torch.enable_grad():
            grad_value = value.clone().requires_grad_(True)
            a, b = candidate(grad_value), reference(grad_value)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(torch.autograd.grad(a.sum(), grad_value)[0],
                                      torch.autograd.grad(b.sum(), grad_value)[0], rtol=0, atol=0)
        assert tuple(candidate.state_dict()) == keys
        for k, v in candidate.state_dict().items():
            assert v.dtype == torch.float32
            torch.testing.assert_close(v, weights[k], rtol=0, atol=0)
        candidate.half(); reference.half()
        if device == 'cuda':
            torch.testing.assert_close(candidate(value.half()), reference(value.half()), rtol=0, atol=0)
        print('HALF_COMPENSATED_BACKEND device=' + device + ' passed=true', flush=True)
    # This reduction shape was absent from the earlier two shape batches.
    value = torch.randn(2560, 5120, device='cuda')
    weight = torch.randn(1280, 5120, device='cuda') * 0.02
    bias = torch.randn(1280, device='cuda')
    baseline = lambda x: compensated_linear(x, weight, bias)
    proposed = lambda x: half_compensated_linear(x, weight, bias, block_m=64)
    actual = proposed(value)
    precise = torch.nn.functional.linear(value[:16].double(), weight.double(), bias.double())
    relative = float((actual[:16].double()-precise).norm()/precise.norm())
    assert torch.isfinite(actual).all() and relative < 3e-6
    before = graph_seconds(baseline, value, 15, 3)
    after = graph_seconds(proposed, value, 15, 3)
    print('HALF_COMPENSATED_LAST_SHAPE ' + json.dumps(dict(
        shape=[2560, 5120, 1280], relative_to_fp64=relative,
        accepted_s=before, candidate_s=after, speedup=before/after)), flush=True)
    model = torch.nn.Sequential(torch.nn.Linear(640, 640))
    assert install_compensated_linears(model, backend='scaled-fp16') == {'Linear': 1}
    print('HALF_COMPENSATED_BACKEND_CHECK passed=true full_video_quality=unverified', flush=True)


if __name__ == '__main__': main()
