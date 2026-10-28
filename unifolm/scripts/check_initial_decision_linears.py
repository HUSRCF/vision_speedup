"""Verify all three real initial shapes and both captured stage policies."""
from unittest.mock import patch
import torch
import torch.nn.functional as F
from unifolm_wma.utils.initial_decision_linears import install_initial_decision_linears
from unifolm_wma.utils.triton_half_compensated_linear import half_compensated_linear


@torch.no_grad()
def main():
    torch.manual_seed(431)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for inputs, outputs in ((512,512),(512,4096),(2048,512)):
        root = torch.nn.Module()
        root.init_attn = torch.nn.Sequential(torch.nn.Linear(inputs,outputs))
        root.outside = torch.nn.Linear(3,3)
        keys = tuple(root.state_dict())
        assert install_initial_decision_linears(root) == 1
        assert not getattr(root.outside,'_initial_decision_linear_installed',False)
        root.eval().cuda()
        module = root.init_attn[0]
        x = torch.randn(40960,inputs,device='cuda')*.7
        expected = F.linear(x,module.weight,module.bias)
        with patch('unifolm_wma.utils.triton_half_compensated_linear.half_compensated_linear',
                   side_effect=AssertionError('World stage entered approximate kernel')):
            torch.testing.assert_close(module(x),expected,rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32 = True
        decision = module(x)
        expected = half_compensated_linear(x,module.weight,module.bias,block_m=64,
                                          block_k=32,group_m=32)
        torch.testing.assert_close(decision,expected,rtol=0,atol=0)
        sample = [0,20480,40959]
        exact = F.linear(x[sample].double(),module.weight.double(),module.bias.double())
        error = float((decision[sample].double()-exact).norm()/exact.norm())
        assert error < 1e-5, error
        del decision,expected,exact
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):module(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured = module(x)
        x.mul_(.99)
        module.weight.add_(.00001)
        module.bias.mul_(.95)
        expected = module(x)
        graph.replay()
        torch.testing.assert_close(captured,expected,rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32 = False
        expected = F.linear(x,module.weight,module.bias)
        with patch('unifolm_wma.utils.triton_half_compensated_linear.half_compensated_linear',
                   side_effect=AssertionError('Updated world stage entered approximate kernel')):
            torch.testing.assert_close(module(x),expected,rtol=0,atol=0)
        # The captured decision graph retains its stage-specific kernels.
        graph.replay()
        torch.testing.assert_close(captured,half_compensated_linear(x,module.weight,
            module.bias,block_m=64,block_k=32,group_m=32),rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32 = True
        for mode in ('training','grad','autocast','noncontiguous','shape'):
            module.train(mode == 'training')
            value = x.t().contiguous().t() if mode == 'noncontiguous' else x[:35] if mode == 'shape' else x
            with torch.set_grad_enabled(mode == 'grad'):
                with torch.autocast('cuda',enabled=mode=='autocast',dtype=torch.float16):
                    torch.testing.assert_close(module(value),F.linear(value,module.weight,module.bias),
                                               rtol=0,atol=0)
            module.eval()
        assert keys == tuple(root.state_dict())
        assert all(p.dtype == torch.float32 for p in root.parameters())
        print(f'INITIAL_DECISION_LINEAR shape={(40960,inputs,outputs)} '
              f'world_native_bitwise=true graph_input_weight_refresh=true fp64_relative={error}',flush=True)
        torch.backends.cudnn.allow_tf32 = False
        del root,module,x,expected,captured,graph,value
        torch.cuda.empty_cache()
    print('INITIAL_DECISION_LINEAR_CHECK passed=true actual_shapes=3 '
          'world_native_bitwise=true graph_input_weight_refresh=true native_fallbacks=5 '
          'master_weights=fp32 BK=32 full_video_quality=unverified',flush=True)


if __name__ == '__main__':
    main()
