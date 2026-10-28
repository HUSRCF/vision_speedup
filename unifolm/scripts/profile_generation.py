"""Diagnostic only: time one generation's encoding, denoising and decoding."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import torch
import unifolm_wma.utils.inference_runtime as runtime


class GenerationComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--diagnostic-report", type=Path, required=True)
    args, inference_args = parser.parse_known_args()
    if args.diagnostic_report.exists():
        raise FileExistsError(args.diagnostic_report)
    original_configure = runtime.configure_denoiser
    stats = {}

    def configure(model, *positional, **keywords):
        original_configure(model, *positional, **keywords)
        for name in ("encode_first_stage", "decode_first_stage", "apply_model"):
            original = getattr(model, name)

            def timed(*values, _name=name, _original=original, **options):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                before = time.perf_counter()
                start.record()
                output = _original(*values, **options)
                end.record()
                end.synchronize()
                stats.setdefault(_name, []).append({
                    "cuda_s": start.elapsed_time(end) / 1000,
                    "wall_s": time.perf_counter() - before,
                })
                if _name == "decode_first_stage":
                    summary = {key: {"calls": len(rows),
                               "cuda_s": sum(row["cuda_s"] for row in rows),
                               "wall_s": sum(row["wall_s"] for row in rows),
                               "first_cuda_s": rows[0]["cuda_s"],
                               "last_cuda_s": rows[-1]["cuda_s"]}
                               for key, rows in stats.items()}
                    args.diagnostic_report.parent.mkdir(parents=True, exist_ok=True)
                    args.diagnostic_report.write_text(json.dumps({
                        "scope": "one-generation diagnostic, not a competition result",
                        "summary": summary, "calls": stats}, indent=2) + "\n")
                    print("GENERATION_COMPONENTS", json.dumps(summary), flush=True)
                    raise GenerationComplete
                return output

            setattr(model, name, timed)

    runtime.configure_denoiser = configure
    script = Path(__file__).parent / "evaluation/world_model_interaction.py"
    sys.path.insert(0, str(script.parent.resolve()))
    sys.argv = [str(script), *inference_args]
    print("DIAGNOSTIC ONLY: stops after the first complete generation.", flush=True)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except GenerationComplete:
        pass
    finally:
        runtime.configure_denoiser = original_configure


if __name__ == "__main__":
    main()
