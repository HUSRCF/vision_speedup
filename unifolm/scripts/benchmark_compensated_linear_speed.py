"""Idle-GPU entry point for the compensated Linear diagnostic."""
import sys
import os
from benchmark_compensated_linear import main

if __name__ == "__main__":
    sys.argv.append("--benchmark")
    if os.environ.get("DIAGNOSTIC_BLOCK_K"):
        sys.argv.extend(("--block-k", os.environ["DIAGNOSTIC_BLOCK_K"]))
    main()
