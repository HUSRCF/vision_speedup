"""Check QKV hook, three-master refresh, stage fallbacks and graph bits."""
import json
from unittest.mock import patch
import torch
from unifolm_wma.modules.attention import CrossAttention
from unifolm_wma.utils.compensated_qkv import fused_compensated_qkv
from unifolm_wma.utils.triton_compensated_linear import make_forward as precise_forward
from unifolm_wma.utils.initial_bk32_qkv import (
    install_initial_bk32_qkv, try_initial_bk32_qkv)


def identical(a, b):
    assert len(a) == len(b) == 3
    for left, right in zip(a, b):
        assert left.is_contiguous() and right.is_contiguous()
        assert torch.equal(left.view(torch.int32), right.view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(809)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for rows, channels in ((40960, 512),):
        module = CrossAttention(channels, heads=8, dim_head=channels//8).eval().cuda()
        projections = module.to_q, module.to_k, module.to_v
        for p in projections:
            p.weight.copy_(p.weight.half().float())
            p.forward = precise_forward(p, min_matmul_macs=0,
                backend='scaled-fp16', cache_half_weights=True, group_m=32,
                block_k=32, pack_inputs=False)
            p._tf32x3_installed = True
            p._initial_world_linear_installed = True
            p._initial_decision_linear_installed = True
        module._compensated_self_qkv_enabled = False
        diffusion = torch.nn.Module()
        diffusion.init_attn = torch.nn.Module()
        diffusion.init_attn.selected = module
        keys = tuple(module.state_dict())
        pointers = tuple(p.weight.data_ptr() for p in projections)
        value = torch.randn(rows, channels, device='cuda') * .7
        expected = fused_compensated_qkv(module, value)
        assert install_initial_bk32_qkv(diffusion) == 1
        try:
            install_initial_bk32_qkv(diffusion)
            raise AssertionError('Duplicate QKV residual hook accepted')
        except ValueError:
            pass
        identical(fused_compensated_qkv(module, value), expected)
        cache = module._initial_bk32_qkv_cache
        cache_pointers = cache[1].data_ptr(), tuple(p.data_ptr() for p in cache[2]), cache[3].data_ptr()
        assert int(cache[3]) == 0
        assert fused_compensated_qkv(module, value) is not None
        assert module._initial_bk32_qkv_cache is cache
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fused_compensated_qkv(module, value)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = fused_compensated_qkv(module, value)
        for index, p in enumerate(projections):
            value.add_(.01)
            p.weight.add_(.00001)
            eager = fused_compensated_qkv(module, value)
            updated = module._initial_bk32_qkv_cache
            assert cache_pointers == (updated[1].data_ptr(), tuple(t.data_ptr() for t in updated[2]), updated[3].data_ptr())
            assert int(updated[3]) > 0
            graph.replay()
            identical(captured, eager)
            module._initial_bk32_qkv_installed = False
            identical(captured, fused_compensated_qkv(module, value))
            module._initial_bk32_qkv_installed = True
        for p in projections:
            p.weight.copy_(p.weight.half().float())
        eager = fused_compensated_qkv(module, value)
        assert int(module._initial_bk32_qkv_cache[3]) == 0
        graph.replay()
        identical(captured, eager)
        for fallback in ('decision', 'training', 'grad', 'autocast',
                         'noncontiguous', 'other_shape', 'matmul_tf32'):
            module.train(fallback == 'training')
            torch.backends.cudnn.allow_tf32 = fallback == 'decision'
            torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
            x = (value.t().contiguous().t() if fallback == 'noncontiguous' else
                 value[:3] if fallback == 'other_shape' else value)
            with torch.set_grad_enabled(fallback == 'grad'):
                with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                    with patch('unifolm_wma.utils.initial_bk32_qkv.zero_weight_low_linear_bk32',
                               side_effect=AssertionError('Fallback entered world QKV')):
                        assert try_initial_bk32_qkv(module, x) is None
            module.eval()
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        delattr(module, '_initial_bk32_qkv_cache')
        module._initial_bk32_qkv_installed = False
        expected = fused_compensated_qkv(module, value)
        module._initial_bk32_qkv_installed = True
        stream.wait_stream(torch.cuda.current_stream())
        cold = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold, stream=stream):
            cold_result = fused_compensated_qkv(module, value)
        cold.replay()
        identical(cold_result, expected)
        assert keys == tuple(module.state_dict())
        assert pointers == tuple(p.weight.data_ptr() for p in projections)
        assert all(p.dtype == torch.float32 for p in module.parameters())
        print('INITIAL_BK32_QKV_RUNTIME ' + json.dumps(dict(shape=[rows, channels],
            previous_qkv_bits=True, each_of_three_master_refresh=True,
            zero_nonzero_zero_graph_input_master_bits=True, cache_parts_predicate_storage_reuse=True,
            lazy_cache=True, fallback_checks=7, cold_capture_previous_bits=True,
            master_state_storage_preserved=True)), flush=True)
        del graph, cold, module, value, expected, eager, captured, cold_result, cache, updated
        torch.cuda.empty_cache()
    print('INITIAL_BK32_QKV_RUNTIME_CHECK passed=true configs=1 '
          'full_video_quality=unverified official_score=false', flush=True)


if __name__ == '__main__':
    main()
