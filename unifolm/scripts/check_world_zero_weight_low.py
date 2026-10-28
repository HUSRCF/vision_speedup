"""Test lazy residual detection and captured zero/nonzero weight refresh."""
import json
from unittest.mock import patch
import torch
from unifolm_wma.utils.triton_compensated_linear import make_forward as precise_forward
from unifolm_wma.utils.world_zero_weight_low_linear import make_forward, install_world_zero_weight_low_linears
from unifolm_wma.utils.triton_zero_weight_low_linear import weight_low_count


def identical(a, b):
    assert torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(787)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for rows, inputs in ((2560, 1280), (640, 1280), (2560, 5120), (640, 5120)):
        for bias in (False, True):
            root = torch.nn.Module()
            root.selected = torch.nn.Linear(inputs, 1280, bias=bias)
            root.other = torch.nn.Linear(inputs, 640, bias=bias)
            root.eval().cuda()
            module = root.selected
            module.weight.copy_(module.weight.half().float())
            module.forward = precise_forward(module, min_matmul_macs=0,
                backend='scaled-fp16', cache_half_weights=True, group_m=32,
                block_k=64, pack_inputs=True)
            module._tf32x3_installed = True
            root.other._tf32x3_installed = True
            original, other = module.forward, root.other.forward
            keys = tuple(root.state_dict())
            pointer = module.weight.data_ptr()
            assert install_world_zero_weight_low_linears(root) == 1
            assert root.other.forward == other
            try:
                install_world_zero_weight_low_linears(root)
                raise AssertionError('Duplicate zero residual install accepted')
            except ValueError:
                pass
            value = torch.randn(rows, inputs, device='cuda') * .7
            predicates = []
            def record_count(parts, *, out=None):
                count = weight_low_count(parts, out=out)
                predicates.append((count.data_ptr(), int(count)))
                return count
            with patch('unifolm_wma.utils.world_zero_weight_low_linear.weight_low_count',
                       side_effect=record_count):
                identical(module(value), original(value))
                module(value)
                assert len(predicates) == 1 and predicates[0][1] == 0
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        module(value)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = module(value)
                for index, exact in enumerate((False, True), 2):
                    value.add_(.01)
                    module.weight.add_(.00001)
                    if exact:
                        module.weight.copy_(module.weight.half().float())
                    if module.bias is not None:
                        module.bias.mul_(.95)
                    eager = module(value)
                    assert len(predicates) == index
                    assert predicates[-1][0] == predicates[0][0]
                    assert (predicates[-1][1] == 0) == exact
                    graph.replay()
                    identical(captured, eager)
                    identical(captured, original(value))
            for fallback in ('decision', 'training', 'grad', 'autocast',
                             'noncontiguous', 'other_shape', 'matmul_tf32'):
                module.train(fallback == 'training')
                torch.backends.cudnn.allow_tf32 = fallback == 'decision'
                torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
                x = (value.t().contiguous().t() if fallback == 'noncontiguous' else
                     value[:3] if fallback == 'other_shape' else value)
                with torch.set_grad_enabled(fallback == 'grad'):
                    with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                        with patch('unifolm_wma.utils.world_zero_weight_low_linear.zero_weight_low_linear',
                                   side_effect=AssertionError('Fallback entered new linear')):
                            identical(module(x), original(x))
                module.eval()
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
            module.forward = original
            module.forward = make_forward(module)
            expected = original(value)
            stream.wait_stream(torch.cuda.current_stream())
            cold = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cold, stream=stream):
                cold_result = module(value)
            cold.replay()
            identical(cold_result, expected)
            assert keys == tuple(root.state_dict()) and module.weight.data_ptr() == pointer
            assert all(p.dtype == torch.float32 for p in root.parameters())
            print('WORLD_ZERO_WEIGHT_LOW_RUNTIME ' + json.dumps(dict(shape=[rows, 1280, inputs],
                bias=bias, short3half_bits=True, zero_nonzero_zero_graph_bits=True,
                lazy_weight_predicate_cache=True, version_predicate_storage_reuse=True,
                other_modules_excluded=True, native_fallbacks=7, cold_capture_previous_bits=True,
                master_state_storage_preserved=True)), flush=True)
            del graph, cold, root, module, value, eager, captured, cold_result, expected
            torch.cuda.empty_cache()
    print('WORLD_ZERO_WEIGHT_LOW_RUNTIME_CHECK passed=true configs=8 '
          'full_video_quality=unverified official_score=false', flush=True)


if __name__ == '__main__':
    main()
