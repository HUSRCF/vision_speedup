"""Diagnostic only: compare packed-head and native BMHK xFormers layouts.

Timing includes the layout copies required by the current inference path.
Projection weights, precision, attention scale, and masks are unchanged.
"""
import argparse
import json

import torch
import xformers.ops

from benchmark_temporal_conv2d import graph_seconds


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--accuracy-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    print("DIAGNOSTIC_ONLY attention_layout; not a competition result", flush=True)
    batch, dim = 16, 64
    for queries, heads in ((2560, 5), (640, 10), (160, 20), (40, 20)):
        for keys, masked in ((queries, False), (77, False), (16, False),
                             (2, False), (256, True)):
            q = torch.randn(batch, queries, heads * dim, device="cuda")
            k = torch.randn(batch, keys, heads * dim, device="cuda")
            v = torch.randn_like(k)
            bias3 = None
            if masked:
                columns = torch.arange(keys, device="cuda")
                limits = (torch.arange(batch, device="cuda") + 1) * 16
                mask = columns.unsqueeze(0) >= limits.unsqueeze(1)
                base = torch.zeros(batch, keys, device="cuda").masked_fill_(
                    mask, float("-inf"))
                bias3 = base.repeat_interleave(heads, 0).unsqueeze(1).expand(
                    batch * heads, queries, keys)
            bias4 = (None if bias3 is None else
                     bias3.reshape(batch, heads, queries, keys))

            def pack(value):
                return value.reshape(batch, value.shape[1], heads, dim).permute(
                    0, 2, 1, 3).reshape(batch * heads, value.shape[1], dim).contiguous()

            def native(query):
                output = xformers.ops.memory_efficient_attention(
                    pack(query), pack(k), pack(v), attn_bias=bias3)
                return output.reshape(batch, heads, queries, dim).permute(
                    0, 2, 1, 3).reshape(batch, queries, heads * dim)

            def bmhk(query):
                output = xformers.ops.memory_efficient_attention(
                    query.reshape(batch, queries, heads, dim),
                    k.reshape(batch, keys, heads, dim),
                    v.reshape(batch, keys, heads, dim), attn_bias=bias4)
                return output.reshape(batch, queries, heads * dim)

            reference, result = native(q), bmhk(q)
            difference = result - reference
            accuracy = {
                "batch": batch, "queries": queries, "keys": keys,
                "heads": heads, "masked": masked,
                "bitwise_equal": torch.equal(result, reference),
                "max_absolute": difference.abs().max().item(),
                "relative_l2": (difference.norm() / reference.norm()).item(),
            }
            print("ATTENTION_LAYOUT_ACCURACY", json.dumps(accuracy), flush=True)
            if args.accuracy_only:
                del q, k, v, reference, result, difference, bias3, bias4
                torch.cuda.empty_cache()
                continue
            native_s = graph_seconds(native, q, args.iterations, args.repeats)
            bmhk_s = graph_seconds(bmhk, q, args.iterations, args.repeats)
            print("ATTENTION_LAYOUT_BENCHMARK", json.dumps({
                **accuracy, "native_s": native_s, "bmhk_s": bmhk_s,
                "speedup": native_s / bmhk_s,
                "saving_s": native_s - bmhk_s,
            }), flush=True)
            del q, k, v, reference, result, difference, bias3, bias4
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
