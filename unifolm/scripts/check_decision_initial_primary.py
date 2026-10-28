"""Validate initial512 precision scope and measure all three actual shapes."""
import json
from unittest.mock import patch
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.initial_decision_linears import install_initial_decision_linears
from unifolm_wma.utils.initial_world_linears import install_initial_world_linears
from unifolm_wma.utils.decision_initial_primary import install_decision_initial_primary, make_forward
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_primary_narrow_linear import primary_narrow_linear
from unifolm_wma.utils.decision_primary_ffn import primary_linear


@torch.no_grad()
def main():
    torch.manual_seed(773)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    for scope, count in [('projections', 8), ('ffn', 2), ('all', 10)]:
        root = torch.nn.Module()
        root.init_attn = torch.nn.ModuleList([torch.nn.Linear(512, 512) for _ in range(8)] +
            [torch.nn.Linear(512, 4096), torch.nn.Linear(2048, 512)])
        root.unrelated = torch.nn.Linear(512, 512)
        assert install_initial_decision_linears(root) == 10
        assert install_initial_world_linears(root) == 10
        assert install_decision_initial_primary(root, scope) == count
        assert not getattr(root.unrelated, '_decision_initial_primary_installed', False)
        for layer in root.init_attn:
            expected = scope == 'all' or ((layer.in_features, layer.out_features) == (512,512)) == (scope == 'projections')
            assert bool(getattr(layer, '_decision_initial_primary_installed', False)) == expected
    for inputs, outputs, bias in [(512,512,False),(512,512,True),
                                  (512,4096,True),(2048,512,True)]:
        root = torch.nn.Module()
        root.init_attn = torch.nn.Sequential(torch.nn.Linear(inputs, outputs, bias=bias))
        root.eval().cuda()
        assert install_initial_decision_linears(root) == 1
        assert install_initial_world_linears(root) == 1
        module = root.init_attn[0]
        original = module.forward
        keys = tuple(root.state_dict())
        assert install_decision_initial_primary(root, 'all') == 1
        value = torch.randn(40960, inputs, device='cuda') * .7
        torch.backends.cudnn.allow_tf32 = False
        with patch('unifolm_wma.utils.decision_initial_primary.primary_narrow_linear',
                   side_effect=AssertionError('World entered initial512 primary')):
            torch.testing.assert_close(module(value), original(value), rtol=0, atol=0)
        torch.backends.cudnn.allow_tf32 = True
        result = module(value)
        parts = pack_half_weight(module.weight)
        poisoned = primary_narrow_linear(value, parts[0], module.bias,
            tile=(64,128,32,4), debug_poison=True)
        torch.testing.assert_close(result, poisoned, rtol=0, atol=0)
        reference = primary_linear(value, parts[0], module.bias, debug_poison=True)
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=2e-6)
        sample = [0, 20000, 40959]
        exact = F.linear(value[sample].double(), module.weight.double(),
                         None if module.bias is None else module.bias.double())
        relative = float((result[sample].double()-exact).norm()/exact.norm())
        assert relative < 6e-4 and torch.isfinite(result).all(), relative
        before = graph_seconds(original, value, 5, 3)
        seconds = graph_seconds(module, value, 5, 3)
        after = graph_seconds(original, value, 5, 3)
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
        for fallback in ('training','grad','autocast','noncontiguous','shape','matmul_tf32'):
            module.train(fallback == 'training')
            torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
            x = (value.t().contiguous().t() if fallback == 'noncontiguous' else
                 value[:35] if fallback == 'shape' else value)
            with torch.set_grad_enabled(fallback == 'grad'):
                with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                    with patch('unifolm_wma.utils.decision_initial_primary.primary_narrow_linear',
                               side_effect=AssertionError('Fallback entered initial512 primary')):
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
        assert keys == tuple(root.state_dict()) and all(p.dtype == torch.float32 for p in root.parameters())
        print('DECISION_INITIAL_PRIMARY ' + json.dumps(dict(shape=[40960,inputs,outputs],
            bias=bias, fp64_relative=relative, before_s=before, after_s=after,
            candidate_s=seconds, conservative_speedup=min(before,after)/seconds,
            cublas_primary_bitwise=bool(torch.equal(result, reference)))), flush=True)
    print('DECISION_INITIAL_PRIMARY_CHECK passed=true actual_shapes=3 bias_branches=4 '
          'named_membership=true world_previous_bitwise=true poison_exact=true '
          'cublas_primary_close=true native_fallbacks=6 cold_capture_fallback=true '
          'graph_input_weight_refresh=true master_bias_accum_output=fp32 full_video_quality=unverified',flush=True)


if __name__ == '__main__':
    main()
