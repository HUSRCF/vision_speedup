"""Measure repeated conditioning work in one real FP32 denoiser call.

This diagnostic runs the ordinary evaluation entry point and exits after two
warm-up calls plus one measured call. It does not write a benchmark result.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

import torch
import unifolm_wma.utils.inference_runtime as runtime


class ProfileComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--diagnostic-report", type=Path, required=True)
    args, inference_args = parser.parse_known_args()
    if args.diagnostic_report.exists():
        raise FileExistsError(args.diagnostic_report)

    original_configure = runtime.configure_denoiser
    measured_call = 3

    def configure(model, *positional, **keywords):
        original_configure(model, *positional, **keywords)
        diffusion = model.model.diffusion_model
        action = diffusion.action_unet
        state = diffusion.state_unet
        events = {}
        wrapped = set()

        def wrap(module, label):
            if id(module) in wrapped:
                return
            wrapped.add(id(module))
            original = module.forward

            def timed(*values, **options):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = original(*values, **options)
                end.record()
                events.setdefault(label, []).append((start, end))
                return result

            module.forward = timed

        # Wrap children before parents. Nested CUDA event ranges are expected;
        # the report exposes both total head time and its repeated sub-parts.
        wrap(action.obs_encoder, "action_obs_encoder")
        for module in action.spatial_softmax_blocks:
            wrap(module, "action_spatial_softmax")
        wrap(state.obs_encoder, "state_obs_encoder")
        for module in state.spatial_softmax_blocks:
            wrap(module, "state_spatial_softmax")
        wrap(diffusion.action_token_projector, "action_token_projector")
        # In the video backbone, query projections depend on the current noisy
        # latent while cross-attention key/value projections depend only on the
        # fixed conditioning for a 50-step generation.
        for name, module in diffusion.named_modules():
            if name.startswith(("action_unet.", "state_unet.")):
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf == "to_q":
                wrap(module, "video_query_projection")
            elif leaf in {"to_k", "to_v", "to_k_ip", "to_v_ip",
                          "to_k_as", "to_v_as", "to_k_aa", "to_v_aa"}:
                wrap(module, "video_condition_kv_projection")
        wrap(action, "action_head_total")
        wrap(state, "state_head_total")
        wrap(diffusion, "complete_denoiser")

        original_apply = model.apply_model
        calls = 0

        def measured(*values, **options):
            nonlocal calls
            calls += 1
            offsets = {name: len(rows) for name, rows in events.items()}
            result = original_apply(*values, **options)
            if calls == measured_call:
                torch.cuda.synchronize()
                times = {}
                invocations = {}
                for name, rows in events.items():
                    selected = rows[offsets.get(name, 0):]
                    times[name] = sum(start.elapsed_time(end)
                                      for start, end in selected) / 1000
                    invocations[name] = len(selected)
                total = times["complete_denoiser"]
                heads = times["action_head_total"] + times["state_head_total"]
                times["video_backbone_inferred"] = total - heads
                report = {
                    "scope": "third eager FP32 denoiser call; diagnostic only",
                    "call": calls,
                    "seconds": times,
                    "invocations": invocations,
                    "fractions_of_complete": {
                        name: value / total for name, value in times.items()
                    },
                }
                args.diagnostic_report.parent.mkdir(parents=True, exist_ok=True)
                args.diagnostic_report.write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("CONDITIONING_PROFILE " + json.dumps(report), flush=True)
                raise ProfileComplete
            return result

        model.apply_model = measured

    runtime.configure_denoiser = configure
    script = Path(__file__).parent / "evaluation/world_model_interaction.py"
    sys.path.insert(0, str(script.parent.resolve()))
    sys.argv = [str(script), *inference_args]
    print("DIAGNOSTIC ONLY: exits after the third denoiser call.", flush=True)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except ProfileComplete:
        pass
    finally:
        runtime.configure_denoiser = original_configure


if __name__ == "__main__":
    main()
