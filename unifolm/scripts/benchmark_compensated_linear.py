"""Numeric/performance diagnostic for the fused three-product Linear kernel."""

import argparse
import json
from pathlib import Path

import torch

from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.tf32x3 import make_forward
from unifolm_wma.utils.triton_compensated_linear import compensated_linear


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--ptx", type=Path)
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for rows, inputs, outputs in (
            (10240, 640, 5120), (10240, 2560, 640),
            (2560, 5120, 1280), (10240, 640, 640),
            (2560, 1280, 1280), (2560, 1280, 10240),
            (640, 1280, 1280), (640, 1280, 10240),
            (640, 5120, 1280)):
        module = torch.nn.Linear(inputs, outputs).cuda().eval()
        value = torch.randn(rows, inputs, device="cuda")
        reference = torch.nn.functional.linear(
            value[:16].double(), module.weight.double(), module.bias.double())
        baseline = module(value)
        old_result = make_forward(module)(value)
        candidate = lambda x: compensated_linear(
            x, module.weight, module.bias, block_k=args.block_k)
        actual, kernel = compensated_linear(
            value, module.weight, module.bias, block_k=args.block_k,
            return_kernel=True)
        errors = {label: float((result[:16].double() - reference).norm() /
                               reference.norm()) for label, result in
                  (("fp32", baseline), ("torch_three_product", old_result),
                   ("triton_three_product", actual))}
        record = dict(shape=[rows, inputs, outputs], block_k=args.block_k,
                      relative_l2_vs_fp64=errors,
                      max_abs_vs_fp32=float((actual - baseline).abs().max()))
        assert torch.isfinite(actual).all()
        if args.benchmark:
            baseline_s = graph_seconds(module, value, 20, 5)
            candidate_s = graph_seconds(candidate, value, 20, 5)
            record.update(baseline_s=baseline_s, candidate_s=candidate_s,
                          speedup=baseline_s / candidate_s)
        print("COMPENSATED_LINEAR_CHECK " + json.dumps(record), flush=True)
        if args.ptx and not args.ptx.exists():
            args.ptx.write_text(kernel.asm["ptx"])
        del module, value, actual, old_result, baseline, reference, candidate
        torch.cuda.empty_cache()
    print("COMPENSATED_LINEAR_CHECK completed=true full_video_quality=unverified",
          flush=True)


if __name__ == "__main__":
    main()
