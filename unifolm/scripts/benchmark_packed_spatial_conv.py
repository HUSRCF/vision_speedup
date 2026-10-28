"""Actual deep spatial shapes; include patch split and deterministic reduction."""
import json
from unittest.mock import patch
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_packed_spatial_conv import packed_spatial_conv, spatial_weight_parts


def sampled_error(value, weight, bias, result):
    exact = F.conv2d(value[:1].double(), weight[:8].double(),
                    None if bias is None else bias[:8].double(), padding=1)
    return float((result[:1, :8].double() - exact).norm() / exact.norm())


@torch.no_grad()
def main():
    torch.manual_seed(577)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    rows = []
    for inputs in (1280, 2560):
        module = torch.nn.Conv2d(inputs, 1280, 3, padding=1).cuda().eval()
        value = torch.randn(16, inputs, 5, 8, device='cuda') * .7
        reference = module(value)
        parts = spatial_weight_parts(module.weight)
        for bm, bn, splits in [(m, n, s) for m in (32, 64) for n in (64, 128) for s in (1, 2, 4)]:
            def candidate(x):
                return packed_spatial_conv(x, parts, module.bias,
                    block_m=bm, block_n=bn, splits=splits)
            result = candidate(value)
            error = sampled_error(value, module.weight, module.bias, result)
            assert torch.isfinite(result).all() and error < 3e-6, error
            with patch('torch.empty', wraps=torch.empty) as allocation:
                # Poison all fully written intermediate/output allocations.
                def poisoned(*args, **kwargs):
                    return torch.full(*args, fill_value=float('nan'), **kwargs)
                allocation.side_effect = poisoned
                torch.testing.assert_close(candidate(value), result, rtol=0, atol=0)
            before = graph_seconds(module, value, 5, 3)
            seconds = graph_seconds(candidate, value, 5, 3)
            after = graph_seconds(module, value, 5, 3)
            row = dict(shape=list(value.shape), config=[bm, bn, splits], fp64_relative=error,
                       native_relative=float((result - reference).norm() / reference.norm()),
                       before_s=before, candidate_s=seconds, after_s=after,
                       conservative_speedup=min(before, after) / seconds,
                       all_pack_reduce_layout_in_timing=True, poisoned_bitwise=True)
            rows.append(row)
            print('PACKED_SPATIAL_CONV ' + json.dumps(row), flush=True)
        best = max((r for r in rows if r['shape'][1] == inputs), key=lambda r:r['conservative_speedup'])
        bm, bn, splits = best['config']
        def chosen(x):
            return packed_spatial_conv(x, parts, module.bias,
                block_m=bm, block_n=bn, splits=splits)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                chosen(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = chosen(value)
        value.add_(.01)
        module.weight.add_(.00001)
        module.bias.mul_(.95)
        spatial_weight_parts(module.weight, out=parts)
        eager = chosen(value)
        graph.replay()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        print('PACKED_SPATIAL_CONV_SELECTED ' + json.dumps(best), flush=True)
        del module, value, reference, parts, result, captured, eager, graph
        torch.cuda.empty_cache()
    x = torch.randn(2, 37, 7, 9, device='cuda') * .7
    w = torch.randn(70, 37, 3, 3, device='cuda') * .02
    parts = spatial_weight_parts(w)
    for splits in (1, 2, 4):
        for bias in (None, torch.randn(70, device='cuda') * .02):
            result = packed_spatial_conv(x, parts, bias, block_m=32, block_n=64, splits=splits)
            assert sampled_error(x, w, bias, result) < 3e-6
    print('PACKED_SPATIAL_CONV_CHECK passed=true actual_shapes=2 configs=12 '
          'poisoned_outputs=true deterministic_split_reduce=true odd_tails_bias_branches=true '
          'graph_input_weight_refresh=true inference_installed=false full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
