#!/usr/bin/env python3
"""Compare xFormers FMHA with explicit BMM for real spatial-attention shapes."""
import argparse
import statistics

import torch
import xformers.ops


CASES = (
    # (batch*heads, query tokens, key tokens, head dimension)
    (128, 2560, 2, 64),
    (128, 2560, 16, 64),
    (128, 640, 2, 64),
    (128, 640, 16, 64),
    (128, 640, 77, 64),
    (128, 160, 2, 64),
    (128, 160, 16, 64),
    (128, 160, 77, 64),
)

SELF_CASES = (
    (128, 2560, 64),
    (128, 640, 64),
    (128, 160, 64),
)


def elapsed_ms(function, iterations):
    for _ in range(5):
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
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = True
    for batch_heads, tokens, dim in SELF_CASES:
        q = torch.randn(batch_heads, tokens, dim, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def fmha():
            return xformers.ops.memory_efficient_attention(q, k, v)

        def fmha_bf16():
            return xformers.ops.memory_efficient_attention(
                q.to(torch.bfloat16), k.to(torch.bfloat16),
                v.to(torch.bfloat16)).float()

        def fmha_fp16():
            return xformers.ops.memory_efficient_attention(
                q.to(torch.float16), k.to(torch.float16),
                v.to(torch.float16)).float()

        reference = fmha()
        bf16_difference = (reference - fmha_bf16()).abs()
        fp16_difference = (reference - fmha_fp16()).abs()
        iterations = max(5, args.iterations // 3)
        fmha_ms = elapsed_ms(fmha, iterations)
        bf16_ms = elapsed_ms(fmha_bf16, iterations)
        fp16_ms = elapsed_ms(fmha_fp16, iterations)
        print(
            f"self=true bh={batch_heads} n={tokens} d={dim} "
            f"fmha_ms={fmha_ms:.4f} "
            f"bf16_max_error={bf16_difference.max().item():.9g} "
            f"bf16_mean_error={bf16_difference.mean().item():.9g} "
            f"bf16_ms={bf16_ms:.4f} bf16_speedup={fmha_ms / bf16_ms:.3f} "
            f"fp16_max_error={fp16_difference.max().item():.9g} "
            f"fp16_mean_error={fp16_difference.mean().item():.9g} "
            f"fp16_ms={fp16_ms:.4f} fp16_speedup={fmha_ms / fp16_ms:.3f}",
            flush=True)
    for batch_heads, queries, keys, dim in CASES:
        q = torch.randn(batch_heads, queries, dim, device="cuda")
        k = torch.randn(batch_heads, keys, dim, device="cuda")
        v = torch.randn_like(k)

        def fmha():
            return xformers.ops.memory_efficient_attention(q, k, v)

        def fmha_bf16():
            return xformers.ops.memory_efficient_attention(
                q.to(torch.bfloat16), k.to(torch.bfloat16),
                v.to(torch.bfloat16)).float()

        def fmha_fp16():
            return xformers.ops.memory_efficient_attention(
                q.to(torch.float16), k.to(torch.float16),
                v.to(torch.float16)).float()

        def bmm():
            scores = torch.bmm(q, k.transpose(1, 2))
            scores.mul_(dim ** -0.5)
            return torch.bmm(scores.softmax(dim=-1), v)

        reference = fmha()
        actual_bf16 = fmha_bf16()
        difference_bf16 = (reference - actual_bf16).abs()
        actual_fp16 = fmha_fp16()
        difference_fp16 = (reference - actual_fp16).abs()
        actual = bmm()
        difference = (reference - actual).abs()
        fmha_ms = elapsed_ms(fmha, args.iterations)
        fmha_bf16_ms = elapsed_ms(fmha_bf16, args.iterations)
        fmha_fp16_ms = elapsed_ms(fmha_fp16, args.iterations)
        bmm_ms = elapsed_ms(bmm, args.iterations)
        torch.backends.cuda.matmul.allow_tf32 = False
        actual_fp32 = bmm()
        difference_fp32 = (reference - actual_fp32).abs()
        bmm_fp32_ms = elapsed_ms(bmm, args.iterations)
        torch.backends.cuda.matmul.allow_tf32 = True
        print(
            f"bh={batch_heads} q={queries} k={keys} d={dim} "
            f"max_error={difference.max().item():.9g} "
            f"mean_error={difference.mean().item():.9g} "
            f"bf16_max_error={difference_bf16.max().item():.9g} "
            f"bf16_mean_error={difference_bf16.mean().item():.9g} "
            f"bf16_ms={fmha_bf16_ms:.4f} "
            f"bf16_speedup={fmha_ms / fmha_bf16_ms:.3f} "
            f"fp16_max_error={difference_fp16.max().item():.9g} "
            f"fp16_mean_error={difference_fp16.mean().item():.9g} "
            f"fp16_ms={fmha_fp16_ms:.4f} "
            f"fp16_speedup={fmha_ms / fmha_fp16_ms:.3f} "
            f"fmha_ms={fmha_ms:.4f} bmm_ms={bmm_ms:.4f} "
            f"bmm_speedup={fmha_ms / bmm_ms:.3f} "
            f"fp32_max_error={difference_fp32.max().item():.9g} "
            f"fp32_mean_error={difference_fp32.mean().item():.9g} "
            f"bmm_fp32_ms={bmm_fp32_ms:.4f} "
            f"bmm_fp32_speedup={fmha_ms / bmm_fp32_ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
