"""Check real CrossAttention dispatch, graph refresh and native fallbacks."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from benchmark_compensated_attention import sampled_error
from unifolm_wma.modules.attention import CrossAttention
from unifolm_wma.utils.triton_compensated_attention import compensated_attention


def main():
    directory = Path(os.environ['ATTENTION_FIXTURE_DIR'])
    fixture = torch.load(directory / 'qkv-00.pt', map_location='cpu')
    q, k, v = (fixture[name].cuda().contiguous()
               for name in ('query', 'key', 'value'))
    module = CrossAttention(query_dim=320, heads=5, dim_head=64).cuda().eval()
    module._compensated_spatial_attention = True
    keys = list(module.state_dict())
    with torch.no_grad():
        actual = module._efficient_attention(q, k, v)
        expected = compensated_attention(q, k, v, block_m=64, block_n=64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert sampled_error(q, k, v, actual) < 1e-5
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                module._efficient_attention(q, k, v)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = module._efficient_attention(q, k, v)
        q.add_(.1)
        k.mul_(.9)
        v.sub_(.15)
        graph.replay()
        torch.testing.assert_close(captured, module._efficient_attention(q, k, v),
                                   rtol=0, atol=0)
    # A sentinel native op checks dispatch without asking unsupported CPU/mask
    # cases to execute a CUDA-only native kernel.
    with patch('xformers.ops.memory_efficient_attention',
               side_effect=lambda a, *args, **kwargs: a) as native:
        def fallback(a=q, b=k, c=v, **kwargs):
            before = native.call_count
            module._efficient_attention(a, b, c, **kwargs)
            assert native.call_count == before + 1
        with torch.no_grad():
            module._compensated_spatial_attention = False
            fallback()
            module._compensated_spatial_attention = True
            module.train()
            fallback()
            module.eval()
            fallback(attn_bias=torch.zeros(1, device='cuda'))
            fallback(q[:, :10], k[:, :10], v[:, :10])
            fallback(q.transpose(1, 2).contiguous().transpose(1, 2), k, v)
            fallback(q.half(), k.half(), v.half())
            fallback(*(fixture[n] for n in ('query', 'key', 'value')))
            module._inference_projection_dtype = torch.float16
            fallback()
            module._inference_projection_dtype = torch.float32
        with torch.enable_grad():
            fallback()
    assert list(module.state_dict()) == keys
    assert all(parameter.dtype == torch.float32 for parameter in module.parameters())
    print('COMPENSATED_ATTENTION_RUNTIME_CHECK ' + json.dumps(dict(
        passed=True, actual_class=True, fp64_sample=True, cuda_graph_refresh=True,
        native_fallbacks=9, fp32_master_weights=True, state_keys_unchanged=True,
        full_video_quality='unverified')), flush=True)


if __name__ == '__main__':
    main()
