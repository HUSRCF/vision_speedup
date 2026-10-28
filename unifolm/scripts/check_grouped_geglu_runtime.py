"""Exercise launch grouping through the real legacy GEGLU module wrapper."""
import os
import torch
from unifolm_wma.modules.attention import GEGLU
from unifolm_wma.utils.triton_compensated_geglu import compensated_geglu, make_forward


@torch.no_grad()
def main():
    torch.manual_seed(353)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    os.environ['UNIFOLM_COMPENSATED_GEGLU_GROUP_M'] = '16'
    module = GEGLU(320, 1280).eval().cuda()
    native = module.forward
    keys = tuple(module.state_dict())
    module.forward = make_forward(module)
    x = torch.randn(16, 2560, 320, device='cuda') * .7
    expected = compensated_geglu(x, module.proj.weight, module.proj.bias)
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = module(x)
    x.add_(.01)
    module.proj.weight.add_(.0001)
    module.proj.bias.mul_(.95)
    graph.replay()
    torch.testing.assert_close(captured, module(x), rtol=0, atol=0)
    expected = compensated_geglu(x, module.proj.weight, module.proj.bias)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    for case in ('training', 'grad', 'noncontiguous', 'rows'):
        module.train(case == 'training')
        value = x.transpose(0, 1) if case == 'noncontiguous' else x[:1] if case == 'rows' else x
        with torch.set_grad_enabled(case == 'grad'):
            torch.testing.assert_close(module(value), native(value), rtol=0, atol=0)
        module.eval()
    assert tuple(module.state_dict()) == keys
    assert all(p.dtype == torch.float32 for p in module.parameters())
    cpu = GEGLU(320, 1280).eval()
    original = cpu.forward
    cpu.forward = make_forward(cpu)
    value = torch.randn(1, 320)
    torch.testing.assert_close(cpu(value), original(value), rtol=0, atol=0)
    print('GROUPED_GEGLU_RUNTIME_CHECK passed=true group_m=16 real_shape=true '
          'bitwise=true graph_input_weight_refresh=true native_fallbacks=5 '
          'master_weights=fp32 state_keys_unchanged=true full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
