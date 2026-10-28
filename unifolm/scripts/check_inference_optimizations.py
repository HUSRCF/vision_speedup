"""Small GPU equivalence checks against the untouched official source.

Run after sourcing scripts/team07_env.sh with one otherwise idle visible GPU.
Full video PSNR validation remains necessary after these operator checks.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch

from unifolm_wma.modules.attention import CrossAttention
from unifolm_wma.models.samplers.ddim import schedule_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=Path(
        "/share/problems/unifolm-world-model-action"))
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "official_attention", args.reference_root /
        "src/unifolm_wma/modules/attention.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda", 0)
    settings = dict(query_dim=128, context_dim=128, heads=2,
                    dim_head=64, image_cross_attention=True, video_length=16,
                    agent_state_context_len=2, agent_action_context_len=256)
    original = reference.CrossAttention(**settings).eval().to(device)
    optimized = CrossAttention(**settings).eval().to(device)
    optimized.load_state_dict(original.state_dict(), strict=True)

    with torch.no_grad():
        for queries in (33, 64, 160):
            x = torch.randn(16, queries, 128, device=device)
            for context_length in (None, 93, 95, 351):
                context = None if context_length is None else torch.randn(
                    16, context_length, 128, device=device)
                expected = original(x, context)
                actual = optimized(x, context)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # Exercise the cache hit with new input, not a repeated output.
                x_next = torch.randn_like(x)
                torch.testing.assert_close(optimized(x_next, context),
                                           original(x_next, context),
                                           rtol=0, atol=0)

        dense = original._get_attn_mask_aa(16, 160, 256).to(device)
        dense = dense.unsqueeze(1).repeat(1, 2, 1, 1).reshape(32, 160, 256)
        compact = optimized._get_cached_attn_mask_aa(
            16, 160, 256, device=device, dtype=torch.float32)
        torch.testing.assert_close(compact, dense, rtol=0, atol=0)
        assert compact.stride(1) == 0
        assert compact.untyped_storage().nbytes() < dense.untyped_storage().nbytes()
        print("attention: 24 comparisons bitwise equal; mask storage bytes:",
              dense.untyped_storage().nbytes(), "->",
              compact.untyped_storage().nbytes(), flush=True)

        values = torch.linspace(0.001, 0.999, 50, device=device)
        for shape in ((1, 1, 1, 1), (3, 1, 1, 1, 1)):
            for index in (0, 25, 49):
                expected = torch.full(shape, values[index], device=device)
                actual = schedule_value(values, index, shape, device)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert actual.dtype == expected.dtype
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            schedule_value(values, 25, (1, 1, 1, 1, 1), device)
        names = {event.key for event in prof.key_averages()}
        assert "aten::_local_scalar_dense" not in names, names
        print("schedule: values/dtype equal; no host scalar extraction", flush=True)
        if args.benchmark:
            settings.update(query_dim=320, context_dim=1024, heads=5)
            original = reference.CrossAttention(**settings).eval().to(device)
            optimized = CrossAttention(**settings).eval().to(device)
            optimized.load_state_dict(original.state_dict(), strict=True)
            x = torch.randn(16, 2560, 320, device=device)
            context = torch.randn(16, 351, 1024, device=device)
            torch.testing.assert_close(optimized(x, context),
                                       original(x, context), rtol=0, atol=0)
            times = {"official": [], "cached_mask": []}
            for _ in range(5):
                for label, module in (("official", original),
                                      ("cached_mask", optimized)):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    module(x, context)
                    torch.cuda.synchronize()
                    times[label].append(time.perf_counter() - started)
            print("ATTENTION_MICROBENCH", json.dumps({
                "scope": "single cross-attention layer; not end-to-end",
                "median_seconds": {key: statistics.median(value)
                                   for key, value in times.items()},
                "samples_seconds": times,
            }), flush=True)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
