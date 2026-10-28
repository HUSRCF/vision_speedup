"""Guard stages, shape routing, FP32 masters and graph freshness for 1x1."""
from unittest.mock import patch
import torch
from unifolm_wma.utils.pointwise_nchw_runtime import SELECTED,install_pointwise_nchw,make_forward
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_pointwise_nchw import pointwise_nchw


@torch.no_grad()
def main():
    torch.manual_seed(503)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    for shape,(bm,group) in SELECTED.items():
        batch,inner,columns,height,width=shape
        module=torch.nn.Conv2d(inner,columns,1).eval().cuda()
        original=module.forward
        keys=tuple(module.state_dict())
        value=torch.randn(batch,inner,height,width,device='cuda')*.7
        original(value)
        assert install_pointwise_nchw(module)==1
        result=module(value)
        parts=pack_half_weight(module.weight.view(columns,inner))
        expected=pointwise_nchw(value,parts,module.bias,block_m=bm,group_m=group)
        torch.testing.assert_close(result,expected,rtol=0,atol=0)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):module(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=module(value)
        value.add_(.01)
        module.weight.add_(.00001)
        module.bias.mul_(.95)
        eager=module(value)
        graph.replay()
        torch.testing.assert_close(captured,eager,rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32=True
        with patch('unifolm_wma.utils.pointwise_nchw_runtime.pointwise_nchw',
                   side_effect=AssertionError('Decision entered custom pointwise')):
            torch.testing.assert_close(module(value),original(value),rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32=False
        if shape==(16,320,640,20,32):
            for fallback in ('training','grad','autocast','noncontiguous','shape','matmul_tf32'):
                module.train(fallback=='training')
                torch.backends.cuda.matmul.allow_tf32=fallback=='matmul_tf32'
                x=value.transpose(-1,-2) if fallback=='noncontiguous' else value[:1] if fallback=='shape' else value
                with torch.set_grad_enabled(fallback=='grad'):
                    with torch.autocast('cuda',enabled=fallback=='autocast',dtype=torch.float16):
                        with patch('unifolm_wma.utils.pointwise_nchw_runtime.pointwise_nchw',
                                   side_effect=AssertionError('Fallback entered custom pointwise')):
                            torch.testing.assert_close(module(x),original(x),rtol=0,atol=0)
                module.eval()
                torch.backends.cuda.matmul.allow_tf32=False
            module.forward=original
            module.forward=make_forward(module)
            cold_expected=original(value)
            cold=torch.cuda.CUDAGraph()
            with torch.cuda.graph(cold,stream=stream):cold_result=module(value)
            cold.replay()
            torch.testing.assert_close(cold_result,cold_expected,rtol=0,atol=0)
            del cold,cold_result,cold_expected,x
        assert keys==tuple(module.state_dict())
        assert all(p.dtype==torch.float32 for p in module.parameters())
        print(f'POINTWISE_NCHW_RUNTIME shape={shape} decision_native_bitwise=true '
              'graph_input_weight_refresh=true master_weights=fp32',flush=True)
        del module,original,value,result,parts,expected,graph,captured,eager
        torch.cuda.empty_cache()
    print('POINTWISE_NCHW_RUNTIME_CHECK passed=true actual_shapes=10 '
          'decision_native_bitwise=true native_fallbacks=6 cold_capture_fallback=true '
          'graph_input_weight_refresh=true master_weights=fp32 full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
