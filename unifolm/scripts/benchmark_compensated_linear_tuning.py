"""Check and tune launch geometry without changing the active full-case run."""

import argparse
import json

import torch

from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_compensated_linear import compensated_linear


CONFIGS = (
    dict(block_m=32, block_n=64, block_k=32, group_m=0, num_warps=4),
    dict(block_m=32, block_n=64, block_k=32, group_m=8, num_warps=4),
    dict(block_m=32, block_n=64, block_k=32, group_m=16, num_warps=4),
    dict(block_m=64, block_n=64, block_k=32, group_m=8, num_warps=4),
    dict(block_m=32, block_n=128, block_k=32, group_m=8, num_warps=4),
    dict(block_m=64, block_n=128, block_k=32, group_m=8, num_warps=8),
)
SHAPES = (
    (10240, 640, 5120), (10240, 2560, 640), (2560, 5120, 1280),
    (10240, 640, 640), (2560, 1280, 1280), (2560, 1280, 10240),
    (640, 1280, 1280), (640, 1280, 10240), (640, 5120, 1280),
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for rows, inputs, outputs in SHAPES:
        value = torch.randn(rows, inputs, device="cuda")
        weight = torch.randn(outputs, inputs, device="cuda")
        bias = torch.randn(outputs, device="cuda")
        reference = compensated_linear(value, weight, bias)
        native = lambda x: torch.nn.functional.linear(x, weight, bias)
        native_s = graph_seconds(native, value, 10, 3) if args.benchmark else None
        for config in CONFIGS:
            candidate = lambda x: compensated_linear(x, weight, bias, **config)
            actual, kernel = compensated_linear(value, weight, bias,
                                                 return_kernel=True, **config)
            difference = actual - reference
            record = dict(shape=[rows, inputs, outputs], config=config,
                          bitwise_equal=torch.equal(actual, reference),
                          max_abs=float(difference.abs().max()),
                          relative_l2=float(difference.norm() / reference.norm()),
                          registers=getattr(kernel, "n_regs", None),
                          spills=getattr(kernel, "n_spills", None))
            # Grouping and changing M/N tiles must preserve the arithmetic
            # used by the full-quality candidate (same K=32 rounding blocks).
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            if args.benchmark:
                elapsed = graph_seconds(candidate, value, 10, 3)
                record.update(native_s=native_s, candidate_s=elapsed,
                              speedup=native_s / elapsed)
            print("COMPENSATED_LINEAR_TUNING " + json.dumps(record), flush=True)
            del actual, difference, candidate
        del value, weight, bias, reference, native
        torch.cuda.empty_cache()
    print("COMPENSATED_LINEAR_TUNING completed=true", flush=True)


if __name__ == "__main__":
    main()
