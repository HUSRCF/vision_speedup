"""Exercise actual FFN shapes, world protection, graph refresh and fallbacks."""
import os
from unittest.mock import patch
import torch
import torch.nn.functional as F
from unifolm_wma.utils.triton_compensated_linear import make_forward as precise_forward
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.decision_vendor_ffn import install_decision_vendor_ffns,make_forward
from unifolm_wma.utils.cublas_compensated_linear import cublas_compensated_linear
from unifolm_wma.utils.cublas_chunked_compensated_linear import chunked_compensated_linear


@torch.no_grad()
def main():
    mode = os.environ['UNIFOLM_DECISION_VENDOR_FFN']
    assert mode in {'full','chunked'}
    torch.manual_seed(467)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for rows,inner,columns in ((10240,2560,640),(2560,5120,1280),(640,5120,1280)):
        root = torch.nn.Module()
        root.block = torch.nn.Module()
        root.block.ff = torch.nn.Module()
        root.block.ff.net = torch.nn.Sequential(torch.nn.Identity(),torch.nn.Identity(),
                                              torch.nn.Linear(inner,columns))
        root.outside = torch.nn.Linear(3,3)
        root.eval().cuda()
        module = root.block.ff.net[2]
        module.forward = precise_forward(module,backend='scaled-fp16',min_matmul_macs=0,
            cache_half_weights=True,group_m=32,block_k=64,pack_inputs=True)
        module._tf32x3_installed = True
        original = module.forward
        keys = tuple(root.state_dict())
        assert install_decision_vendor_ffns(root,mode)==1
        assert not getattr(root.outside,'_decision_vendor_ffn_installed',False)
        value = torch.randn(rows,inner,device='cuda')*.7
        with patch('unifolm_wma.utils.decision_vendor_ffn.cublas_compensated_linear',
                   side_effect=AssertionError('World entered vendor')),patch(
                   'unifolm_wma.utils.decision_vendor_ffn.chunked_compensated_linear',
                   side_effect=AssertionError('World entered vendor')):
            torch.testing.assert_close(module(value),original(value),rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32 = True
        def reference():
            parts = pack_half_weight(module.weight)
            if mode=='full':return cublas_compensated_linear(value,parts,module.bias)
            return chunked_compensated_linear(value,parts,module.bias,
                chunk_k=512 if inner==2560 else 1024)
        result = module(value)
        torch.testing.assert_close(result,reference(),rtol=0,atol=0)
        selected = [0,rows//2,rows-1]
        exact = F.linear(value[selected].double(),module.weight.double(),module.bias.double())
        error = float((result[selected].double()-exact).norm()/exact.norm())
        assert torch.isfinite(result).all() and error<2e-5,error
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):module(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured = module(value)
        value.add_(.01)
        module.weight.add_(.00001)
        module.bias.mul_(.95)
        eager = module(value)
        graph.replay()
        torch.testing.assert_close(captured,eager,rtol=0,atol=0)
        torch.testing.assert_close(captured,reference(),rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32 = False
        torch.testing.assert_close(module(value),original(value),rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32 = True
        if rows==640:
            for fallback in ('training','grad','autocast','noncontiguous','shape'):
                module.train(fallback=='training')
                x = value.t().contiguous().t() if fallback=='noncontiguous' else value[:35] if fallback=='shape' else value
                with torch.set_grad_enabled(fallback=='grad'):
                    with torch.autocast('cuda',enabled=fallback=='autocast',dtype=torch.float16):
                        with patch('unifolm_wma.utils.decision_vendor_ffn.cublas_compensated_linear',side_effect=AssertionError('Fallback vendor')),patch('unifolm_wma.utils.decision_vendor_ffn.chunked_compensated_linear',side_effect=AssertionError('Fallback vendor')):
                            torch.testing.assert_close(module(x),original(x),rtol=0,atol=0)
                module.eval()
            module.forward = original
            module.forward = make_forward(module,mode)
            expected = original(value)
            cold = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cold,stream=stream):cold_result = module(value)
            cold.replay()
            torch.testing.assert_close(cold_result,expected,rtol=0,atol=0)
            del cold,cold_result,x,expected
        assert tuple(root.state_dict())==keys
        assert all(p.dtype==torch.float32 for p in root.parameters())
        print(f'DECISION_VENDOR_FFN_RUNTIME mode={mode} shape={(rows,inner,columns)} '
              f'fp64_relative={error} world_bitwise=true graph_input_weight_refresh=true',flush=True)
        torch.backends.cudnn.allow_tf32 = False
        del root,module,value,result,exact,eager,captured,graph,original
        torch.cuda.empty_cache()
    print(f'DECISION_VENDOR_FFN_CHECK passed=true mode={mode} actual_shapes=3 '
          'world_native_bitwise=true cold_capture_fallback=true '
          'graph_input_weight_refresh=true native_fallbacks=5 master_weights=fp32 '
          'full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
