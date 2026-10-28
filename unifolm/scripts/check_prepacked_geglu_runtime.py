"""Check real wider GEGLU routing, FP32 state, fallbacks and graph freshness."""
from unittest.mock import patch
import torch
from unifolm_wma.modules.attention import GEGLU
from unifolm_wma.utils.triton_compensated_linear import make_forward as projection_forward
from unifolm_wma.utils.triton_prepacked_geglu import install_prepacked_geglus


@torch.no_grad()
def main():
    torch.manual_seed(443)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for rows,channels in ((10240,640),(2560,1280),(640,1280)):
        module = GEGLU(channels,4*channels).eval().cuda()
        module.proj.forward = projection_forward(module.proj,backend='scaled-fp16',
            min_matmul_macs=0,cache_half_weights=True,group_m=32,block_k=64,pack_inputs=True)
        module.proj._tf32x3_installed = True
        native = module.forward
        keys = tuple(module.state_dict())
        x = torch.randn(rows,channels,device='cuda')*.7
        expected = native(x)
        assert install_prepacked_geglus(module)==1
        torch.testing.assert_close(module(x),expected,rtol=2e-6,atol=2e-6)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):module(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured = module(x)
        x.add_(.01)
        module.proj.weight.add_(.00001)
        module.proj.bias.mul_(.95)
        eager = module(x)
        graph.replay()
        torch.testing.assert_close(captured,eager,rtol=0,atol=0)
        torch.testing.assert_close(eager,native(x),rtol=2e-6,atol=2e-6)
        if rows==640:
            for mode in ('training','grad','autocast','noncontiguous','shape'):
                module.train(mode=='training')
                value = x.t().contiguous().t() if mode=='noncontiguous' else x[:35] if mode=='shape' else x
                with torch.set_grad_enabled(mode=='grad'):
                    with torch.autocast('cuda',enabled=mode=='autocast',dtype=torch.float16):
                        with patch('unifolm_wma.utils.triton_prepacked_geglu.prepacked_geglu',
                                   side_effect=AssertionError('Fallback entered fused GEGLU')):
                            torch.testing.assert_close(module(value),native(value),rtol=0,atol=0)
                module.eval()
            del value
        assert keys==tuple(module.state_dict())
        assert all(p.dtype==torch.float32 for p in module.parameters())
        print(f'PREPACKED_GEGLU_RUNTIME shape={(rows,channels)} '
              'graph_input_weight_refresh=true master_weights=fp32',flush=True)
        del module,x,expected,eager,captured,graph,native
        torch.cuda.empty_cache()
    print('PREPACKED_GEGLU_RUNTIME_CHECK passed=true actual_shapes=3 '
          'graph_input_weight_refresh=true native_fallbacks=5 master_weights=fp32 '
          'BK=64 full_video_quality=unverified',flush=True)


if __name__=='__main__':
    main()
