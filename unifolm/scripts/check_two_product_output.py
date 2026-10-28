"""Arithmetic, live graph and stage guards before any full precision experiment."""
import json
from unittest.mock import patch
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_compensated_linear import make_forward as precise_forward
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight, half_compensated_linear
from unifolm_wma.utils.triton_two_product_linear import two_product_linear
from unifolm_wma.utils.world_two_product_output import install_world_two_product_output, make_forward
from unifolm_wma.utils.decision_primary_narrow import install_decision_primary_narrow


def identical(left, right):
    assert torch.equal(left.view(torch.int32), right.view(torch.int32))


def oracle(value, weight, bias, protect):
    x_high = value.half().double()
    x_low = ((value - value.half().float()) * 2048).half().double()
    w_high = (weight * 256).half().double()
    w_low = ((weight * 256 - (weight * 256).half().float()) * 2048).half().double()
    x_effective = x_high + x_low / 2048 if protect == 'input' else x_high
    w_effective = w_high / 256 if protect == 'input' else (w_high + w_low / 2048) / 256
    return F.linear(x_effective, w_effective, None if bias is None else bias.double())


@torch.no_grad()
def main():
    torch.manual_seed(643)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Include non-multiple tiles and both bias branches in addition to actual workload.
    for shape in [(40960, 320, 320), (131, 73, 79), (17, 65, 31)]:
        m, n, k = shape
        for bias_enabled in (False, True):
            value = torch.randn(m, k, device='cuda') * .7
            weight = torch.randn(n, k, device='cuda') * .03
            bias = torch.randn(n, device='cuda') * .01 if bias_enabled else None
            parts = pack_half_weight(weight)
            sample = list(dict.fromkeys([0, m // 2, m - 1]))
            for protect in ('input', 'weight'):
                full_exact = F.linear(value[sample].double(), weight.double(),
                                       None if bias is None else bias.double())
                def candidate(x):
                    return two_product_linear(x, parts, bias, protect=protect)
                result = candidate(value)
                assert torch.isfinite(result).all()
                identical(result, two_product_linear(value, parts, bias,
                                                     protect=protect, debug_poison=True))
                effective_exact = oracle(value[sample], weight, bias, protect)
                effective_relative = float((result[sample].double() - effective_exact).norm()
                                           / effective_exact.norm())
                full_relative = float((result[sample].double() - full_exact).norm()
                                      / full_exact.norm())
                assert effective_relative < 3e-6, effective_relative
                assert full_relative < 6e-4, full_relative
                timing = None
                if m == 40960:
                    def baseline(x):
                        return half_compensated_linear(x, weight, bias, weight_parts=parts,
                            block_m=64, block_n=64, group_m=32, block_k=64)
                    # Warm both graphs before collecting alternating baseline/candidate timings.
                    graph_seconds(baseline, value, 30, 1)
                    graph_seconds(candidate, value, 30, 1)
                    pairs = []
                    for index in range(6):
                        order = [('baseline', baseline), ('candidate', candidate)]
                        if index % 2:
                            order.reverse()
                        pair = {name: graph_seconds(fn, value, 15, 3) for name, fn in order}
                        pairs.append(pair)
                    import statistics
                    base_s = statistics.median(p['baseline'] for p in pairs)
                    new_s = statistics.median(p['candidate'] for p in pairs)
                    timing = dict(baseline_s=base_s, candidate_s=new_s,
                                  speedup=base_s / new_s, alternating_pairs=pairs)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        candidate(value)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = candidate(value)
                value.add_(.01)
                weight.add_(.00001)
                if bias is not None:
                    bias.mul_(.95)
                pack_half_weight(weight, out=parts)
                eager = candidate(value)
                graph.replay()
                identical(captured, eager)
                print('TWO_PRODUCT_KERNEL ' + json.dumps(dict(shape=shape, bias=bias_enabled,
                    protect=protect, effective_fp64_relative=effective_relative,
                    original_fp64_relative=full_relative, timing=timing,
                    poison_full_output=True, graph_input_weight_bias_refresh=True)), flush=True)
                del graph, captured, eager, result
    for protect in ('input', 'weight'):
        root = torch.nn.Module()
        root.block = torch.nn.Module()
        root.block.proj_in = torch.nn.Linear(320, 320)
        root.block.proj_out = torch.nn.Linear(320, 320)
        root.attn = torch.nn.Module()
        for role in ('to_q', 'to_k', 'to_v'):
            setattr(root.attn, role, torch.nn.Linear(320, 320, bias=False))
        root.attn.to_out = torch.nn.Sequential(torch.nn.Linear(320, 320))
        root.unrelated = torch.nn.Linear(320, 320)
        root.eval().cuda()
        for name, module in root.named_modules():
            if isinstance(module, torch.nn.Linear) and name != 'unrelated':
                module.forward = precise_forward(module, min_matmul_macs=0,
                    backend='scaled-fp16', cache_half_weights=True, group_m=32,
                    block_k=64, pack_inputs=False)
                module._tf32x3_installed = True
        assert install_decision_primary_narrow(root, 'all') == 6
        originals = {name: module.forward for name, module in root.named_modules()
                     if isinstance(module, torch.nn.Linear)}
        keys = tuple(root.state_dict())
        assert install_world_two_product_output(root, protect) == 3
        for name, module in root.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            selected = name.endswith(('.to_out.0', '.proj_in', '.proj_out'))
            assert bool(getattr(module, '_world_two_product_output_installed', False)) == selected
            if not selected:
                # Native bound methods are reconstructed on attribute access.
                assert module.forward == originals[name]
        try:
            install_world_two_product_output(root, protect)
            raise AssertionError('Duplicate installation accepted')
        except ValueError:
            pass
        layer = root.attn.to_out[0]
        original = originals['attn.to_out.0']
        value = torch.randn(40960, 320, device='cuda') * .7
        torch.backends.cudnn.allow_tf32 = True
        with patch('unifolm_wma.utils.world_two_product_output.two_product_linear',
                   side_effect=AssertionError('Decision entered world experiment')):
            identical(layer(value), original(value))
        torch.backends.cudnn.allow_tf32 = False
        packed = []
        def record_pack(weight, *, out=None):
            result = pack_half_weight(weight, out=out)
            packed.append(tuple(p.data_ptr() for p in result))
            return result
        with patch('unifolm_wma.utils.world_two_product_output.pack_half_weight',
                   side_effect=record_pack):
            result = layer(value)
            parts = pack_half_weight(layer.weight)
            identical(result, two_product_linear(value, parts, layer.bias, protect=protect))
            assert len(packed) == 1
            layer(value)
            assert len(packed) == 1
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    layer(value)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = layer(value)
            value.add_(.01)
            layer.weight.add_(.00001)
            layer.bias.mul_(.95)
            eager = layer(value)
            assert len(packed) == 2 and packed[0] == packed[1]
            graph.replay()
            identical(captured, eager)
        for fallback in ('training', 'grad', 'autocast', 'noncontiguous', 'shape', 'matmul_tf32'):
            layer.train(fallback == 'training')
            torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
            x = (value.t().contiguous().t() if fallback == 'noncontiguous'
                 else value[:35] if fallback == 'shape' else value)
            with torch.set_grad_enabled(fallback == 'grad'):
                with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                    with patch('unifolm_wma.utils.world_two_product_output.two_product_linear',
                               side_effect=AssertionError('Fallback entered world experiment')):
                        identical(layer(x), original(x))
            layer.eval()
            torch.backends.cuda.matmul.allow_tf32 = False
        layer.forward = original
        layer.forward = make_forward(layer, protect)
        expected = original(value)
        stream.wait_stream(torch.cuda.current_stream())
        cold = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold, stream=stream):
            cold_result = layer(value)
        cold.replay()
        identical(cold_result, expected)
        assert tuple(root.state_dict()) == keys
        assert all(p.dtype == torch.float32 for p in root.parameters())
        print('TWO_PRODUCT_RUNTIME ' + json.dumps(dict(protect=protect,
            selected_outputs=3, qkv_unrelated_untouched=True, decision_bits=True,
            version_refresh_storage_reused=True, graph_refresh_bits=True,
            native_fallbacks=6, cold_capture_fallback=True, master_state_unchanged=True)), flush=True)
    print('TWO_PRODUCT_CHECK passed=true kernel_configs=12 runtime_modes=2 '
          'full_trajectory_quality=unverified official_score=false', flush=True)


if __name__ == '__main__':
    main()
