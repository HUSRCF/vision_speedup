"""Check masked tiles, inference guards and graph replay for the new GEMM."""

import torch

from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_compensated_linear import (
    compensated_linear, make_forward, install_compensated_linears)


def main():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with torch.no_grad():
        for shape in ((5, 7, 13), (35, 37, 70)):
            rows, inputs, outputs = shape
            value = torch.randn(rows, inputs, device="cuda")
            weight = torch.randn(outputs, inputs, device="cuda")
            for bias in (None, torch.randn(outputs, device="cuda")):
                actual = compensated_linear(value, weight, bias)
                expected = torch.nn.functional.linear(
                    value.double(), weight.double(),
                    None if bias is None else bias.double())
                torch.testing.assert_close(actual.double(), expected,
                                           rtol=2e-5, atol=3e-6)
                # Also verify graph replay on dimensions requiring tile masks.
                graph_seconds(lambda x: compensated_linear(x, weight, bias),
                              value, 2, 1)

        module = torch.nn.Linear(640, 640).eval()
        original = module.forward
        keys = set(module.state_dict())
        wrapped = make_forward(module, min_matmul_macs=0)
        cpu_value = torch.randn(2, 640)
        torch.testing.assert_close(wrapped(cpu_value), original(cpu_value),
                                   rtol=0, atol=0)
        module.cuda()
        noncontiguous = torch.randn(640, 4, device="cuda").t()
        assert not noncontiguous.is_contiguous()
        torch.testing.assert_close(wrapped(noncontiguous), original(noncontiguous),
                                   rtol=0, atol=0)
        value = torch.randn(2, 640, device="cuda")
        module.train()
        torch.testing.assert_close(wrapped(value), original(value), rtol=0, atol=0)
        module.eval().half()
        torch.testing.assert_close(wrapped(value.half()), original(value.half()),
                                   rtol=0, atol=0)
        assert set(module.state_dict()) == keys

    # Grad-enabled execution must preserve the original autograd path.
    module.float()
    value = torch.randn(2, 640, device="cuda", requires_grad=True)
    expected = original(value)
    actual = wrapped(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert value.grad is not None
    model = torch.nn.Sequential(torch.nn.Linear(640, 640),
                                torch.nn.Conv2d(8, 8, 3))
    convolution = model[1].forward
    assert install_compensated_linears(model) == {"Linear": 1}
    assert model[1].forward == convolution
    try:
        install_compensated_linears(model)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Duplicate installation was accepted")
    print("COMPENSATED_LINEAR_GUARDS passed=true: tile masks, bias, graph replay, "
          "CPU, noncontiguous input, training, dtype, autograd, state keys, "
          "convolution preservation, duplicate installation", flush=True)


if __name__ == "__main__":
    main()
