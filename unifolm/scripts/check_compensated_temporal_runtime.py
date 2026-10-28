"""Check actual temporal wrappers, TF32 protection and graph cache refresh."""
from unittest.mock import patch
import torch
from unifolm_wma.utils.temporal_conv2d import make_temporal_conv2d_forward
from unifolm_wma.utils.triton_compensated_temporal_conv import (
    make_compensated_temporal_forward, temporal_weight_parts, compensated_temporal_conv)


@torch.no_grad()
def main():
    torch.manual_seed(419)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    for channels,height,width in ((320,40,64),(640,20,32),(1280,10,16),(1280,5,8)):
        module=torch.nn.Conv3d(channels,channels,(3,1,1),padding=(1,0,0)).eval().cuda()
        keys=tuple(module.state_dict())
        module.forward=make_temporal_conv2d_forward(module,preserve_tf32=True)
        native=module.forward
        x=torch.randn(1,channels,16,height,width,device='cuda')*.7
        native(x)
        module.forward=make_compensated_temporal_forward(module)
        reference=compensated_temporal_conv(x,temporal_weight_parts(module.weight),
            module.bias,block_m=64)
        torch.testing.assert_close(module(x),reference,rtol=0,atol=0)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):module(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=module(x)
        x.add_(.01)
        module.weight.add_(.00001)
        module.bias.mul_(.95)
        expected=module(x)
        graph.replay()
        torch.testing.assert_close(captured,expected,rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32=True
        expected=native(x)
        with patch('unifolm_wma.utils.triton_compensated_temporal_conv.compensated_temporal_conv',
                   side_effect=AssertionError('TF32 decision stage entered compensated kernel')):
            torch.testing.assert_close(module(x),expected,rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32=False
        if height==5:
            module.forward=native
            module.forward=make_compensated_temporal_forward(module)
            expected=native(x)
            cold=torch.cuda.CUDAGraph()
            with torch.cuda.graph(cold,stream=stream):cold_result=module(x)
            cold.replay()
            torch.testing.assert_close(cold_result,expected,rtol=0,atol=0)
            for case in ('training','grad','autocast','noncontiguous','frames','spatial','batch'):
                module.train(case=='training')
                value=(x.transpose(-1,-2) if case=='noncontiguous' else
                    x[:,:,:8].contiguous() if case=='frames' else
                    x[:,:,:,:3,:3].contiguous() if case=='spatial' else
                    x.repeat(2,1,1,1,1) if case=='batch' else x)
                with torch.set_grad_enabled(case=='grad'):
                    with torch.autocast('cuda',enabled=case=='autocast',dtype=torch.float16):
                        with patch('unifolm_wma.utils.triton_compensated_temporal_conv.compensated_temporal_conv',
                                   side_effect=AssertionError('Fallback entered compensated kernel')):
                            torch.testing.assert_close(module(value),native(value),rtol=0,atol=0)
                module.eval()
            del cold,cold_result
        assert tuple(module.state_dict())==keys
        assert all(p.dtype==torch.float32 for p in module.parameters())
        print('COMPENSATED_TEMPORAL_RUNTIME channels='+str(channels)+' spatial='+
            str((height,width))+' graph_input_weight_refresh=true tf32_native_bitwise=true',flush=True)
        del module,x,reference,expected,graph,captured
        torch.cuda.empty_cache()
    cpu=torch.nn.Conv3d(4,4,(3,1,1),padding=(1,0,0)).eval()
    native=cpu.forward
    cpu.forward=make_compensated_temporal_forward(cpu)
    value=torch.randn(1,4,3,2,5)
    torch.testing.assert_close(cpu(value),native(value),rtol=0,atol=0)
    print('COMPENSATED_TEMPORAL_RUNTIME_CHECK passed=true actual_shapes=4 '
          'graph_input_weight_refresh=true cold_native_capture=true '
          'tf32_native_bitwise=true native_fallbacks=8 master_weights=fp32 '
          'full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
