"""Check named precision scope, stage protection and live graph updates."""
import json
from unittest.mock import patch
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_compensated_linear import make_forward as precise_forward
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.decision_primary_ffn import primary_linear
from unifolm_wma.utils.decision_primary_narrow import install_decision_primary_narrow, make_forward


@torch.no_grad()
def main():
    torch.manual_seed(607)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    union = torch.nn.Module()
    union.attn = torch.nn.Module()
    union.attn.to_q = torch.nn.Linear(320, 320, bias=False)
    union.attn.to_out = torch.nn.Sequential(torch.nn.Linear(320, 320))
    union.block = torch.nn.Module()
    union.block.proj_in = torch.nn.Linear(320, 320)
    union.unrelated = torch.nn.Linear(320, 320)
    for selected in (union.attn.to_q, union.attn.to_out[0], union.block.proj_in):
        selected._tf32x3_installed = True
    assert install_decision_primary_narrow(union, 'all') == 3
    assert not getattr(union.unrelated, '_decision_primary_narrow_installed', False)
    for bias in (False, True):
        root = torch.nn.Module()
        root.attn = torch.nn.Module()
        if bias:
            root.attn.to_out = torch.nn.Sequential(torch.nn.Linear(320, 320))
            module = root.attn.to_out[0]
            scope = 'output'
        else:
            root.attn.to_q = torch.nn.Linear(320, 320, bias=False)
            module = root.attn.to_q
            scope = 'qkv'
        root.unrelated = torch.nn.Linear(320, 320)
        root.eval().cuda()
        module.forward = precise_forward(module, min_matmul_macs=0, backend='scaled-fp16',
            cache_half_weights=True, group_m=32, block_k=64, pack_inputs=False)
        module._tf32x3_installed = True
        original = module.forward
        keys = tuple(root.state_dict())
        assert install_decision_primary_narrow(root, scope) == 1
        assert not getattr(root.unrelated, '_decision_primary_narrow_installed', False)
        value = torch.randn(40960, 320, device='cuda') * .7
        torch.backends.cudnn.allow_tf32 = False
        with patch('unifolm_wma.utils.decision_primary_narrow.primary_narrow_linear',
                   side_effect=AssertionError('World entered C320 primary')):
            torch.testing.assert_close(module(value), original(value), rtol=0, atol=0)
        torch.backends.cudnn.allow_tf32 = True
        result = module(value)
        parts = pack_half_weight(module.weight)
        reference = primary_linear(value, parts[0], module.bias, debug_poison=True)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        idx = [0, 20000, 40959]
        exact = F.linear(value[idx].double(), module.weight.double(),
                         None if module.bias is None else module.bias.double())
        error = float((result[idx].double() - exact).norm() / exact.norm())
        assert torch.isfinite(result).all() and error < 6e-4, error
        before = graph_seconds(original, value, 10, 3)
        seconds = graph_seconds(module, value, 10, 5)
        after = graph_seconds(original, value, 10, 3)
        assert min(before, after) > seconds
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
        if module.bias is not None:
            module.bias.mul_(.95)
        eager = module(value)
        graph.replay()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        for fallback in ('training', 'grad', 'autocast', 'noncontiguous', 'shape', 'matmul_tf32'):
            module.train(fallback == 'training')
            torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
            x = (value.t().contiguous().t() if fallback == 'noncontiguous'
                 else value[:35] if fallback == 'shape' else value)
            with torch.set_grad_enabled(fallback == 'grad'):
                with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                    with patch('unifolm_wma.utils.decision_primary_narrow.primary_narrow_linear',
                               side_effect=AssertionError('Fallback entered C320 primary')):
                        torch.testing.assert_close(module(x), original(x), rtol=0, atol=0)
            module.eval()
            torch.backends.cuda.matmul.allow_tf32 = False
        module.forward = original
        module.forward = make_forward(module)
        expected = original(value)
        stream.wait_stream(torch.cuda.current_stream())
        cold = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold, stream=stream):
            cold_result = module(value)
        cold.replay()
        torch.testing.assert_close(cold_result, expected, rtol=0, atol=0)
        assert keys == tuple(root.state_dict())
        assert all(p.dtype == torch.float32 for p in root.parameters())
        print('DECISION_PRIMARY_NARROW ' + json.dumps(dict(scope=scope, bias=bias,
            fp64_relative=error, before_s=before, after_s=after, candidate_s=seconds,
            conservative_speedup=min(before, after) / seconds)), flush=True)
    print('DECISION_PRIMARY_NARROW_CHECK passed=true actual_shape=true named_scope_exclusion=true union_scope=true '
          'world_previous_bitwise=true cublas_primary_bitwise=true native_fallbacks=6 '
          'cold_capture_fallback=true graph_input_weight_refresh=true '
          'master_weights=fp32 full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
