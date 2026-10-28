#!/usr/bin/env python3
"""Validate and microbenchmark the inference-only fused GroupNorm kernel."""
import argparse
import statistics
import sys

import torch
import torch.nn.functional as F

from unifolm_wma.utils.triton_group_norm import (
    fused_framewise_group_norm_silu, fused_group_norm,
    fused_temporal_group_norm_silu)


SHAPES = (
    (16, 320, 40, 64),
    (16, 640, 20, 32),
    (16, 1280, 10, 16),
    (16, 1280, 5, 8),
)
TEMPORAL_SHAPES = (
    (1, 320, 16, 40, 64),
    (1, 640, 16, 20, 32),
    (1, 1280, 16, 10, 16),
)


def elapsed_ms(function, iterations):
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    torch.manual_seed(123)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    worst_max = 0.0
    worst_mean = 0.0
    for shape in SHAPES:
        batch, channels = shape[:2]
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        norm = torch.nn.GroupNorm(32, channels, eps=1e-6).cuda().eval()
        scale = torch.randn(batch, channels * 2, 1, 1, device="cuda")
        scale, shift = scale.chunk(2, dim=1)

        def native():
            return F.silu(norm(x) * (1.0 + scale) + shift)

        def fused():
            return fused_group_norm(
                x, norm, scale=scale, shift=shift, apply_silu=True)

        reference = native()
        actual = fused()
        difference = (reference - actual).abs()
        max_error = difference.max().item()
        mean_error = difference.mean().item()
        worst_max = max(worst_max, max_error)
        worst_mean = max(worst_mean, mean_error)
        native_ms = elapsed_ms(native, args.iterations)
        fused_ms = elapsed_ms(fused, args.iterations)
        print(
            f"shape={shape} max_error={max_error:.9g} "
            f"mean_error={mean_error:.9g} native_ms={native_ms:.4f} "
            f"fused_ms={fused_ms:.4f} speedup={native_ms / fused_ms:.3f}",
            flush=True)
    print(f"worst_max_error={worst_max:.9g} "
          f"worst_mean_error={worst_mean:.9g}", flush=True)
    for shape in TEMPORAL_SHAPES:
        _, channels = shape[:2]
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        norm = torch.nn.GroupNorm(32, channels, eps=1e-5).cuda().eval()

        def native():
            return F.silu(norm(x))

        def fused():
            return fused_framewise_group_norm_silu(x, norm)

        def fused_exact():
            return fused_temporal_group_norm_silu(x, norm)

        reference = native()
        actual = fused()
        difference = (reference - actual).abs()
        max_error = difference.max().item()
        mean_error = difference.mean().item()
        native_ms = elapsed_ms(native, args.iterations)
        fused_ms = elapsed_ms(fused, args.iterations)
        exact = fused_exact()
        exact_difference = (reference - exact).abs()
        exact_ms = elapsed_ms(fused_exact, args.iterations)
        print(
            f"temporal_shape={shape} approximate=true "
            f"max_error={max_error:.9g} mean_error={mean_error:.9g} "
            f"native_ms={native_ms:.4f} fused_ms={fused_ms:.4f} "
            f"speedup={native_ms / fused_ms:.3f} "
            f"exact_max_error={exact_difference.max().item():.9g} "
            f"exact_mean_error={exact_difference.mean().item():.9g} "
            f"exact_ms={exact_ms:.4f} "
            f"exact_speedup={native_ms / exact_ms:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
