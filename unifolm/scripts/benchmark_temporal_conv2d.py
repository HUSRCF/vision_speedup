"""Diagnostic only: compare equivalent temporal Conv3d/Conv2d descriptors."""
import argparse
import json
import statistics

import torch

from unifolm_wma.utils.temporal_conv2d import make_temporal_conv2d_forward


def graph_seconds(function, value, iterations, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = function(value)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, function(value), rtol=0, atol=0)
    measurements = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) / 1000 / iterations)
    return statistics.median(measurements)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--cudnn-tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--preserve-tf32", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cudnn.allow_tf32 = args.cudnn_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.cpu_check:
        for batch, channels, length, height, width, groups in (
                (1, 4, 5, 4, 7, 1), (2, 8, 3, 1, 4, 2)):
            for stride, dilation in ((1, 1), (2, 2)):
                module = torch.nn.Conv3d(
                    channels, channels * 2, (3, 1, 1),
                    stride=(stride, 1, 1), padding=(dilation, 0, 0),
                    dilation=(dilation, 1, 1), groups=groups).double().eval()
                value = torch.randn(batch, channels, length, height, width,
                                    dtype=torch.float64)
                candidate = make_temporal_conv2d_forward(module)
                torch.testing.assert_close(candidate(value), module(value),
                                           rtol=1e-12, atol=1e-12)
                # A non-contiguous spatial view must still map correctly.
                value = value.transpose(-1, -2)
                torch.testing.assert_close(candidate(value), module(value),
                                           rtol=1e-12, atol=1e-12)
        print("TEMPORAL_CONV2D_CPU_CHECK passed=true", flush=True)
        return
    print("DIAGNOSTIC_ONLY temporal_conv2d; not a competition result", flush=True)
    for channels, height, width, count in (
            (320, 40, 64, 20), (640, 20, 32, 20),
            (1280, 10, 16, 20), (1280, 5, 8, 28)):
        module = torch.nn.Conv3d(channels, channels, (3, 1, 1),
                                 padding=(1, 0, 0)).cuda().eval()
        value = torch.randn(1, channels, 16, height, width, device="cuda")
        candidate = make_temporal_conv2d_forward(
            module, preserve_tf32=args.preserve_tf32)
        reference = module(value)
        result = candidate(value)
        difference = result - reference
        relative_l2 = (difference.norm() / reference.norm()).item()
        maximum = difference.abs().max().item()
        print("TEMPORAL_CONV2D_ACCURACY", json.dumps({
            "shape": list(value.shape), "relative_l2": relative_l2,
            "max_absolute": maximum,
            "bitwise_equal": torch.equal(result, reference),
            "cudnn_tf32": args.cudnn_tf32,
            "cudnn_benchmark": args.cudnn_benchmark,
            "preserve_tf32": args.preserve_tf32,
        }), flush=True)
        if args.accuracy_only:
            del module, value, candidate, reference, result, difference
            torch.cuda.empty_cache()
            continue
        native_s = graph_seconds(module, value, args.iterations, args.repeats)
        candidate_s = graph_seconds(candidate, value, args.iterations, args.repeats)
        print("TEMPORAL_CONV2D_BENCHMARK", json.dumps({
            "shape": list(value.shape), "calls_per_denoiser": count,
            "cudnn_tf32": args.cudnn_tf32,
            "cudnn_benchmark": args.cudnn_benchmark,
            "native_s": native_s, "conv2d_s": candidate_s,
            "speedup": native_s / candidate_s,
            "relative_l2": relative_l2, "max_absolute": maximum,
            "bitwise_equal": torch.equal(result, reference),
            "estimated_saving_per_denoiser": (native_s - candidate_s) * count,
        }), flush=True)
        del module, value, candidate, reference, result, difference
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
