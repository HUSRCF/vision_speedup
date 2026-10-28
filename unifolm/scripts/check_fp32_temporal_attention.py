"""Measure IEEE FP32 attention differences and graph speed on actual shapes."""
import copy
import json
import torch
import torch.utils.checkpoint
from einops import rearrange
from unifolm_wma.modules.attention import CrossAttention
from unifolm_wma.utils.triton_temporal_attention_fp32 import fused_temporal_attention_fp32


def native(q, k, v, heads, intermediates=False):
    a, b, c = [rearrange(x, 'b n (h d) -> (b h) n d', h=heads)
               for x in (q, k, v)]
    qk = torch.bmm(a, b.transpose(1, 2))
    probabilities = (qk * .125).softmax(-1)
    output = rearrange(torch.bmm(probabilities, c),
                       '(b h) n d -> b n (h d)', h=heads)
    return (output, qk, probabilities) if intermediates else output


def graph_time(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10): call()
    times = []
    for _ in range(5):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(20): graph.replay()
        end.record(); end.synchronize()
        times.append(start.elapsed_time(end) / 200)
    return sorted(times)[2]


@torch.no_grad()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(123)
    for batch, heads in ((3, 5), (2560, 5), (640, 10), (160, 20), (40, 20)):
        q, k, v = [torch.randn(batch, 16, heads * 64, device='cuda') for _ in range(3)]
        expected = native(q, k, v, heads, True)
        oracle = native(q.double(), k.double(), v.double(), heads, True)
        actual = fused_temporal_attention_fp32(q, k, v, heads, .125, True)
        plain = fused_temporal_attention_fp32(q, k, v, heads, .125)
        torch.testing.assert_close(plain, actual[0], rtol=0, atol=0)
        errors = {}
        for name, a, b, precise in zip(('output', 'qk', 'probabilities'), actual, expected, oracle):
            relative = float((a-b).norm() / b.norm())
            assert torch.isfinite(a).all() and relative < 2e-6
            # FP32 GEMM reduction order differs from cuBLAS. Check accuracy
            # against FP64 rather than requiring cuBLAS's summation tree.
            actual_error = float((a.double()-precise).norm()/precise.norm())
            native_error = float((b.double()-precise).norm()/precise.norm())
            assert actual_error < 2e-6
            assert actual_error <= 4 * native_error + 1e-7
            errors[name] = dict(relative_l2=relative, different=int((a != b).sum()),
                                max_absolute=float((a-b).abs().max()),
                                fused_fp64_relative_error=actual_error,
                                native_fp64_relative_error=native_error)
        before = graph_time(lambda: native(q, k, v, heads))
        after = graph_time(lambda: fused_temporal_attention_fp32(q, k, v, heads, .125))
        print('FP32_TEMPORAL_ATTENTION ' + json.dumps(dict(batch=batch, heads=heads,
              errors=errors, native_ms=before, fused_ms=after, speedup=before/after)), flush=True)
    ref = CrossAttention(320, heads=5, dim_head=64, temporal_length=16).cuda().eval()
    candidate = copy.deepcopy(ref)
    candidate._inference_fused_temporal_attention_fp32 = True
    x = torch.randn(128, 16, 320, device='cuda')
    torch.testing.assert_close(candidate(x), ref(x), rtol=2e-5, atol=2e-6)
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): candidate(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): result = candidate(x)
    x.add_(.1); graph.replay()
    torch.testing.assert_close(result, candidate(x), rtol=0, atol=0)
    # Preserve original TF32-phase, unsupported sequence and Half behavior.
    for dtype, frames, tf32 in ((torch.float32,16,True), (torch.float32,8,False),
                                (torch.float16,16,False)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        baseline = copy.deepcopy(ref).to(dtype)
        test = copy.deepcopy(baseline); test._inference_fused_temporal_attention_fp32 = True
        value = torch.randn(2, frames, 320, device='cuda', dtype=dtype)
        torch.testing.assert_close(test(value), baseline(value), rtol=0, atol=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    print('FP32_TEMPORAL_ATTENTION_CHECK passed=true', flush=True)


if __name__ == '__main__': main()
