"""Diagnostic only: profile one real denoiser call after two warm-up calls.

Pass the ordinary world_model_interaction.py arguments after --diagnostic-dir.
The process stops after profiling, so it produces no competition result.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

import torch

from unifolm_wma.models.ddpms import LatentVisualDiffusion


class ProfileComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    args, inference_args = parser.parse_known_args()
    args.diagnostic_dir.mkdir(parents=True, exist_ok=False)
    print("DIAGNOSTIC ONLY: stops after one profiled denoiser call.", flush=True)
    original = LatentVisualDiffusion.apply_model
    calls = 0

    def profiled(model, *positional, **keywords):
        nonlocal calls
        calls += 1
        if calls < 3:
            return original(model, *positional, **keywords)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            original(model, *positional, **keywords)
            torch.cuda.synchronize()
        averages = prof.key_averages(group_by_input_shape=True)
        records = [{"name": event.key, "count": event.count,
                    "self_cpu_us": event.self_cpu_time_total,
                    "self_cuda_us": event.self_cuda_time_total,
                    "cuda_us": event.cuda_time_total,
                    "shapes": event.input_shapes} for event in averages]
        records.sort(key=lambda item: item["self_cuda_us"], reverse=True)
        (args.diagnostic_dir / "operators.json").write_text(
            json.dumps(records, indent=2) + "\n")
        table = averages.table(sort_by="self_cuda_time_total", row_limit=25)
        (args.diagnostic_dir / "operators.txt").write_text(table)
        print(table, flush=True)
        raise ProfileComplete

    LatentVisualDiffusion.apply_model = profiled
    script = Path(__file__).parent / "evaluation/world_model_interaction.py"
    sys.path.insert(0, str(script.parent.resolve()))
    sys.argv = [str(script), *inference_args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except ProfileComplete:
        print(f"Diagnostic profile saved: {args.diagnostic_dir}", flush=True)
    finally:
        LatentVisualDiffusion.apply_model = original


if __name__ == "__main__":
    main()
