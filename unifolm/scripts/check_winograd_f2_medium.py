"""Confirm 10x16 replacement while preserving the deepest 5x8 native kernels."""
from unittest.mock import patch
import torch
from unifolm_wma.utils.triton_winograd_f2 import (
    f2_spatial_selected, make_f2_forward, transform_filter_f2, winograd_f2_conv2d)


@torch.no_grad()
def main():
    torch.manual_seed(337)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    for scope, expected in [('high', [True, True, True, False]),
                            ('medium', [True, True, False, False]),
                            ('deep', [False, False, True, False])]:
        observed = [f2_spatial_selected(c, hw, scope) for c, hw in
                    [(1280, (10, 16)), (2560, (10, 16)),
                     (1280, (5, 8)), (2560, (5, 8))]]
        assert observed == expected
    for inputs, height, width in ((1280, 10, 16), (2560, 10, 16), (1280, 5, 8)):
        layer = torch.nn.Conv2d(inputs, 1280, 3, padding=1).eval().cuda()
        native = layer.forward
        layer.forward = make_f2_forward(layer, scope='medium')
        x = torch.randn(16, inputs, height, width, device='cuda') * .7
        if height == 5:
            expected = native(x)
            with patch('unifolm_wma.utils.triton_winograd_f2.winograd_f2_conv2d',
                       side_effect=AssertionError('Deep spatial shape must remain native')):
                torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        else:
            expected = winograd_f2_conv2d(x, transform_filter_f2(layer.weight), layer.bias,
                                         block_m=64)
            torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = layer(x)
        x.add_(.01)
        layer.weight.add_(.00001)
        expected = layer(x)
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        print('WINOGRAD_F2_MEDIUM_RUNTIME inputs='+str(inputs)+' spatial='+str((height, width))+
              ' selected='+str(height == 10)+' graph_input_weight_refresh=true', flush=True)
        del layer, x, graph, captured, expected
        torch.cuda.empty_cache()
    print('WINOGRAD_F2_MEDIUM_CHECK passed=true actual_shapes=3 '
          'deep_native_fp32_protected=true graph_input_weight_refresh=true '
          'full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
