"""Check wide gate-only primary products with value branch bitwise protected."""
import json
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_prepacked_geglu import prepacked_geglu
from unifolm_wma.utils.triton_branch_protected_geglu import branch_protected_geglu


@torch.no_grad()
def main():
    torch.manual_seed(719)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for rows, inner, channels, has_bias in [(10240, 640, 2560, True),
                                           (2560, 1280, 5120, True),
                                           (640, 1280, 5120, True),
                                           (35, 173, 645, True),
                                           (35, 173, 645, False)]:
        value = torch.randn(rows, inner, device='cuda') * .7
        weight = torch.randn(channels * 2, inner, device='cuda') * .02
        bias = torch.randn(channels * 2, device='cuda') * .02 if has_bias else None
        parts = pack_half_weight(weight)
        baseline = lambda x: prepacked_geglu(x, parts, bias,
            block_n=64 if inner == 640 else 32, group_m=32)
        control, cv, cg = branch_protected_geglu(value, parts, bias,
            mode='compensated', debug_branches=True)
        torch.testing.assert_close(control, baseline(value), rtol=0, atol=0)
        records = []
        for bn in (32, 64):
            candidate = lambda x: branch_protected_geglu(x, parts, bias,
                mode='gate', block_n=bn, group_m=32)
            result, v, g = branch_protected_geglu(value, parts, bias,
                mode='gate', block_n=bn, group_m=32, debug_branches=True)
            torch.testing.assert_close(v, cv, rtol=0, atol=0)
            torch.testing.assert_close(candidate(value), result, rtol=0, atol=0)
            torch.testing.assert_close(branch_protected_geglu(value, parts, bias,
                mode='gate', block_n=bn, group_m=32, debug_poison=True), result, rtol=0, atol=0)
            idx = [0, rows // 2, rows - 1]
            ev, eg = F.linear(value[idx].double(), weight.double(),
                None if bias is None else bias.double()).chunk(2, dim=-1)
            exact = ev * F.gelu(eg)
            error = float((result[idx].double() - exact).norm()/exact.norm())
            assert error < 1e-3 and torch.isfinite(result).all(), error
            if inner in {640, 1280}:
                before = graph_seconds(baseline, value, 5, 3)
                seconds = graph_seconds(candidate, value, 5, 3)
                after = graph_seconds(baseline, value, 5, 3)
                record = dict(shape=[rows, inner, channels], block_n=bn,
                    fp64_relative=error, before_s=before, after_s=after,
                    candidate_s=seconds, conservative_speedup=min(before, after)/seconds,
                    compensated_control_accepted_bitwise=True,
                    value_branch_accepted_bitwise=True, poisoned_output_bitwise=True)
                records.append(record)
                print('WIDE_GATE_GEGLU ' + json.dumps(record), flush=True)
        bn = max(records, key=lambda r:r['conservative_speedup'])['block_n'] if records else 32
        candidate = lambda x: branch_protected_geglu(x, parts, bias,
            mode='gate', block_n=bn, group_m=32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                candidate(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = candidate(value)
        value.add_(.01)
        weight.add_(.00001)
        if bias is not None:
            bias.mul_(.95)
        pack_half_weight(weight, out=parts)
        graph.replay()
        torch.testing.assert_close(captured, candidate(value), rtol=0, atol=0)
    print('WIDE_GATE_GEGLU_CHECK passed=true actual_shapes=3 configs=6 '
          'compensated_control_accepted_bitwise=true value_branch_accepted_bitwise=true '
          'oddMKN_bias_branches=true poisoned_output=true graph_input_weight_refresh=true '
          'inference_installed=false full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
