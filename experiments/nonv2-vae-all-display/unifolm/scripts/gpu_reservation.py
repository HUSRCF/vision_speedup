#!/usr/bin/env python3
"""Hold a small CUDA allocation while the benchmark process initializes."""

import argparse
import signal
import time
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--ready", required=True, type=Path)
parser.add_argument("--trigger", type=Path)
parser.add_argument("--warmed", type=Path)
parser.add_argument("--mib", type=int, default=64)
args = parser.parse_args()

running = True


def stop(_signum, _frame):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

# Importing torch dominates the cold-start gap.  A waiter may start this
# process while the GPU is busy, then create --trigger when an idle window is
# observed.  No CUDA context exists before that trigger.
if args.warmed is not None:
    args.warmed.write_text(f"pid={__import__('os').getpid()}\n")
if args.trigger is not None:
    while running and not args.trigger.exists():
        time.sleep(0.02)
if not running:
    raise SystemExit(0)

# cudaMalloc makes the reservation visible to nvidia-smi.  The allocation is
# intentionally small: it is a scheduling signal, not a memory denial lock.
reservation = torch.empty(args.mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
args.ready.write_text(f"pid={__import__('os').getpid()} mib={args.mib}\n")

while running:
    time.sleep(0.1)

# Keep the object live until shutdown so the CUDA allocation cannot be freed
# before the benchmark has established its own context.
del reservation
