"""Numerical and CUDA Graph performance checks, never a contest score."""
import json
import os
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_split_group_norm import (
    split_group_norm_silu, install_split_group_norm)


@torch.no_grad()
def main():
    torch.manual_seed(123)
    dtype = (torch.float32 if os.environ.get('SPLIT_NORM_BENCHMARK_DTYPE') == 'fp32'
             else torch.float16)
    for shape in [(16, 320, 40, 64), (16, 640, 20, 32),
                  (16, 1280, 10, 16), (16, 1280, 5, 8),
                  (1, 320, 16, 40, 64), (1, 640, 16, 20, 32),
                  (1, 1280, 16, 10, 16), (1, 1280, 16, 5, 8),
                  (2, 96, 3, 7, 11)]:
        for noncontiguous in (False, True):
            value = torch.randn(shape, device="cuda", dtype=dtype)
            if noncontiguous:
                value = value.transpose(-1, -2)
            norm = torch.nn.GroupNorm(32, shape[1], eps=1e-5).cuda().to(dtype).eval()
            norm.weight.uniform_(.5, 1.5)
            norm.bias.uniform_(-.2, .2)
            native = lambda x: F.silu(norm(x))
            fused = lambda x: split_group_norm_silu(x, norm)
            reference, actual = native(value), fused(value)
            difference = actual.float() - reference.float()
            relative = float(difference.norm() / reference.float().norm())
            assert relative < (.002 if dtype == torch.float16 else 2e-6), relative
            torch.testing.assert_close(actual, reference, rtol=.02, atol=.008)
            native_s = graph_seconds(native, value, 10, 3)
            fused_s = graph_seconds(fused, value, 10, 3)
            print("SPLIT_GROUP_NORM " + json.dumps(dict(
                shape=list(value.shape), dtype=str(dtype), noncontiguous=noncontiguous,
                relative_l2=relative, max_absolute=float(difference.abs().max()),
                exact_fraction=float((actual == reference).float().mean()),
                native_s=native_s, fused_s=fused_s, speedup=native_s / fused_s)),
                flush=True)
    norm = torch.nn.GroupNorm(4, 8).cuda().half().eval()
    value = torch.randn(2, 8, 3, 5, device="cuda", dtype=torch.float16)
    sequence = torch.nn.Sequential(norm, torch.nn.SiLU(), torch.nn.Identity()).eval()
    state_keys = tuple(sequence.state_dict())
    expected = sequence(value)
    assert install_split_group_norm(sequence) == 1
    assert tuple(sequence.state_dict()) == state_keys
    torch.testing.assert_close(sequence(value), expected, rtol=.02, atol=.008)
    sequence.train()
    torch.testing.assert_close(sequence(value), F.silu(norm(value)), rtol=0, atol=0)
    sequence.eval()
    with torch.enable_grad():
        x = value.clone().requires_grad_()
        sequence(x).sum().backward()
        assert x.grad is not None and x.grad.isfinite().all()
    try:
        install_split_group_norm(sequence)
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated installation must fail")
    print("SPLIT_GROUP_NORM_GUARDS passed=true", flush=True)


if __name__ == "__main__":
    main()
