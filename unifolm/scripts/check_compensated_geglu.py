"""Check GEGLU routing, graph refresh, checkpoint keys and native fallback."""
import torch
from unifolm_wma.modules.attention import GEGLU
from unifolm_wma.utils.triton_compensated_geglu import (
    compensated_geglu, make_forward, install_compensated_geglus)


@torch.no_grad()
def main():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    module = GEGLU(320, 1280).cuda().eval()
    original = module.forward
    candidate = make_forward(module)
    value = torch.randn(2560, 16, 320, device='cuda')
    torch.testing.assert_close(candidate(value),
        compensated_geglu(value, module.proj.weight, module.proj.bias), rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            candidate(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = candidate(value)
    value.add_(.1)
    graph.replay()
    torch.testing.assert_close(output, candidate(value), rtol=0, atol=0)
    for x in (value[:1], value.transpose(0, 1).contiguous().transpose(0, 1)):
        torch.testing.assert_close(candidate(x), original(x), rtol=0, atol=0)
    module.train()
    torch.testing.assert_close(candidate(value), original(value), rtol=0, atol=0)
    module.eval()
    with torch.enable_grad():
        x = value[:1].detach().requires_grad_(True)
        a, b = candidate(x), original(x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(a.sum(), x)[0],
            torch.autograd.grad(b.sum(), x)[0], rtol=0, atol=0)
    module.half()
    x = value.half()
    torch.testing.assert_close(candidate(x), original(x), rtol=0, atol=0)
    for inputs, outputs in ((320, 640), (512, 2048)):
        wrong = GEGLU(inputs, outputs).cuda().eval()
        x = torch.randn(40960, inputs, device='cuda')
        torch.testing.assert_close(make_forward(wrong)(x), wrong(x), rtol=0, atol=0)
    model = torch.nn.ModuleList([GEGLU(320, 1280), GEGLU(640, 2560)]).eval()
    keys = tuple(model.state_dict())
    assert install_compensated_geglus(model) == 1
    assert keys == tuple(model.state_dict())
    x = torch.randn(3, 320)
    expected = model[0].activate(model[0].proj(x))
    torch.testing.assert_close(model[0](x), expected, rtol=0, atol=0)
    print('COMPENSATED_GEGLU_GUARDS passed=true full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
