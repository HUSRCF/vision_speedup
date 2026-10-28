"""Verify packed-weight equivalence, invalidation and measured kernel savings."""
import json
import math
import torch
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_half_compensated_linear import (
    half_compensated_linear, pack_half_weight)
from unifolm_wma.utils.triton_compensated_linear import make_forward


@torch.no_grad()
def main():
    torch.manual_seed(157)
    torch.backends.cuda.matmul.allow_tf32 = False
    speeds = []
    shapes = [(40960, 320, 320), (40960, 1280, 320),
              (10240, 640, 640), (10240, 640, 2560), (10240, 2560, 640),
              (2560, 1280, 1280), (2560, 1280, 5120), (2560, 5120, 1280),
              (640, 1280, 1280), (640, 1280, 5120), (640, 5120, 1280),
              (10240, 640, 5120), (2560, 1280, 10240), (640, 1280, 10240),
              (35, 37, 70)]
    for rows, inner, columns in shapes:
        value = torch.randn(rows, inner, device='cuda')
        weight = torch.randn(columns, inner, device='cuda') * .02
        bias = torch.randn(columns, device='cuda')
        parts = pack_half_weight(weight)
        high = (weight * 256).half()
        low = ((weight * 256-high.float()) * 2048).half()
        torch.testing.assert_close(parts[0], high, rtol=0, atol=0)
        torch.testing.assert_close(parts[1], low, rtol=0, atol=0)
        block = 64 if rows >= 2560 else 32
        plain = lambda x: half_compensated_linear(x, weight, bias, block_m=block)
        packed = lambda x: half_compensated_linear(x, weight, bias,
                                                  block_m=block, weight_parts=parts)
        torch.testing.assert_close(packed(value), plain(value), rtol=0, atol=0)
        original_s = graph_seconds(plain, value, 10, 3)
        packed_s = graph_seconds(packed, value, 10, 3)
        speeds.append(original_s/packed_s)
        print('PACKED_HALF_WEIGHT ' + json.dumps(dict(
            shape=[rows, inner, columns], bitwise=True, original_s=original_s,
            packed_s=packed_s, speedup=original_s/packed_s)), flush=True)
        del value, weight, bias, parts, high, low
    module = torch.nn.Linear(37, 70, device='cuda').eval()
    before = {k: v.clone() for k, v in module.state_dict().items()}
    module.forward = make_forward(module, min_matmul_macs=0,
                                  backend='scaled-fp16', cache_half_weights=True)
    value = torch.randn(35, 37, device='cuda')
    # Cold capture may not allocate/cache static weights in Python. Its
    # uncached path must still match the subsequently warmed implementation.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): output = module(value)
    graph.replay()
    torch.testing.assert_close(output, module(value), rtol=0, atol=0)
    for k, v in module.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    with torch.cuda.stream(stream):
        for _ in range(3): module(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): output = module(value)
    # An eager call updates existing packed buffers in place for graph users.
    module.weight.add_(.0001)
    module.bias.mul_(.9)
    value.mul_(.95)
    eager = module(value)
    graph.replay()
    torch.testing.assert_close(output, eager, rtol=0, atol=0)
    expected = half_compensated_linear(value, module.weight, module.bias)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    for mode in ('training', 'grad', 'autocast', 'noncontiguous'):
        if mode == 'training': module.train()
        x = value.t().contiguous().t() if mode == 'noncontiguous' else value
        with torch.set_grad_enabled(mode == 'grad'):
            with torch.autocast('cuda', enabled=mode == 'autocast', dtype=torch.float16):
                torch.testing.assert_close(module(x), torch.nn.functional.linear(
                    x, module.weight, module.bias), rtol=0, atol=0)
        module.eval()
    assert tuple(module.state_dict()) == tuple(before)
    assert all(v.dtype == torch.float32 for v in module.state_dict().values())
    print('PACKED_HALF_WEIGHTS_CHECK ' + json.dumps(dict(
        passed=True, shapes=len(shapes), geometric_speedup=math.exp(
            sum(math.log(s) for s in speeds)/len(speeds)),
        full_video_quality='unverified', warmup_in_timing=True)), flush=True)


if __name__ == '__main__': main()
