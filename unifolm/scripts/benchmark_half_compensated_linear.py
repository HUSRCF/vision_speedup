"""Compare FP16 three-product arithmetic against the accepted TF32 kernel."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_compensated_linear import compensated_linear
from unifolm_wma.utils.triton_half_compensated_linear import half_compensated_linear


@torch.no_grad()
def compare(value, weight, bias, label):
    baseline = lambda x: compensated_linear(x, weight, bias)
    reference = baseline(value)
    rows = value.numel() // value.shape[-1]
    precise = F.linear(value.reshape(rows, -1)[:32].double(), weight.double(),
                       None if bias is None else bias.double())
    before = graph_seconds(baseline, value, 15, 3)
    for bm, bn, warps in ((32, 64, 4), (64, 64, 4), (64, 128, 8)):
        candidate = lambda x: half_compensated_linear(x, weight, bias,
                                     block_m=bm, block_n=bn, num_warps=warps)
        actual = candidate(value)
        error = float((actual.reshape(rows, -1)[:32].double() - precise).norm() / precise.norm())
        delta = float((actual - reference).norm() / reference.norm())
        assert torch.isfinite(actual).all() and error < 3e-6, (label, error)
        after = graph_seconds(candidate, value, 15, 3)
        print('HALF_COMPENSATED_LINEAR ' + json.dumps(dict(
            label=label, shape=[rows, weight.shape[1], weight.shape[0]],
            config=[bm, bn, warps], relative_to_fp64=error, relative_to_accepted=delta,
            accepted_s=before, candidate_s=after, speedup=before/after,
            input_max_abs=float(value.abs().max()), weight_max_abs=float(weight.abs().max()),
            bitwise_accepted=torch.equal(actual, reference))), flush=True)


@torch.no_grad()
def main():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for rows, channels, outputs in ((40960, 1280, 320), (10240, 640, 5120),
                                   (2560, 1280, 10240), (640, 5120, 1280)):
        value = torch.randn(rows, channels, device='cuda')
        weight = torch.randn(outputs, channels, device='cuda') * 0.02
        bias = torch.randn(outputs, device='cuda')
        compare(value, weight, bias, 'synthetic-real-shape')
        del value, weight, bias
        torch.cuda.empty_cache()
    folder = Path.home()/'benchmark-runs/diagnostics/real-narrow-ffn-20261024T220442Z'
    record = json.loads((folder/'fixtures.json').read_text())[0]
    fixture = torch.load(folder/record['file'], map_location='cpu', weights_only=True)
    compare(fixture['value'].cuda(), fixture['weight'].cuda(),
            None if fixture['bias'] is None else fixture['bias'].cuda(), 'actual-layer-input')
    print('HALF_COMPENSATED_CHECK passed=true inference_installed=false full_video_quality=unverified', flush=True)


if __name__ == '__main__': main()
