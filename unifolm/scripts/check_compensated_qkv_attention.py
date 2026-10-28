"""Validate merged QKV inside actual spatial and temporal attention modules."""
import torch
from unittest.mock import patch
from unifolm_wma.modules.attention import CrossAttention
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from unifolm_wma.utils.compensated_qkv import install_compensated_qkv
from unifolm_wma.utils.triton_compensated_linear import install_compensated_linears
from unifolm_wma.utils.triton_compensated_attention import compensated_attention


@torch.no_grad()
def main():
    torch.manual_seed(293)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for channels, batch, length, temporal in (
            (640, 16, 640, False), (1280, 16, 160, False),
            (1280, 160, 16, True), (1280, 40, 16, True)):
        def construct():
            module = CrossAttention(channels, heads=channels//64, dim_head=64,
                temporal_length=16 if temporal else None).eval().cuda()
            module._inference_attention_layout = 'bmhk'
            module._compensated_spatial_attention = not temporal
            module._inference_fused_temporal_attention_fp32 = temporal
            return module
        baseline, candidate = construct(), construct()
        candidate.load_state_dict(baseline.state_dict())
        keys = tuple(candidate.state_dict())
        for module in (baseline, candidate):
            install_compensated_linears(module, backend='scaled-fp16',
                cache_half_weights=True, group_m=32, block_k=64, pack_inputs=True)
        assert install_compensated_qkv(candidate) == 1
        x = torch.randn(batch, length, channels, device='cuda') * .7
        with patch('unifolm_wma.utils.triton_compensated_attention.compensated_attention',
                   wraps=compensated_attention) as attention:
            expected = baseline(x)
            actual = candidate(x)
            if channels == 640 and length == 640:
                assert attention.call_count == 2, 'Merged QKV bypassed accepted attention kernel'
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        wrapper = DenoiserGraph(candidate.forward)
        actual = wrapper(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        x.add_(.01)
        candidate.to_v.weight.add_(.0001)
        baseline.to_v.weight.copy_(candidate.to_v.weight)
        expected = baseline(x)
        candidate(x)  # Refresh timed static weight storage before graph replay.
        actual = wrapper(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert tuple(candidate.state_dict()) == keys
        assert all(p.dtype == torch.float32 for p in candidate.parameters())
        print('COMPENSATED_QKV_ATTENTION shape='+str(tuple(x.shape))+
              ' temporal='+str(temporal)+' bitwise=true graph_input_weight_refresh=true', flush=True)
        del baseline, candidate, wrapper, x, actual, expected
        torch.cuda.empty_cache()
    print('COMPENSATED_QKV_ATTENTION_CHECK passed=true real_modules=4 '
          'spatial_compensated_kernel_retained=true temporal_kernel_retained=true '
          'master_weights=fp32 full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
