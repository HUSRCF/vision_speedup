"""Actual large denoiser Linear shapes, with compensated TF32 arithmetic."""

import argparse
import json

import torch

from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.tf32x3 import make_forward, install_tf32x3


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Installation must leave all convolutions untouched.
    model = torch.nn.Sequential(torch.nn.Linear(640, 640),
                                torch.nn.Conv2d(640, 640, 3),
                                torch.nn.Conv3d(640, 640, (3, 1, 1)))
    original_convs = (model[1].forward, model[2].forward)
    assert install_tf32x3(model, convolutions=False) == {"Linear": 1}
    assert (model[1].forward, model[2].forward) == original_convs
    assert not hasattr(model[1], "_tf32x3_installed")
    del model

    for rows, inputs, outputs in (
            (10240, 640, 640), (10240, 640, 5120),
            (10240, 2560, 640), (2560, 1280, 1280),
            (2560, 1280, 10240), (2560, 5120, 1280),
            (640, 1280, 1280), (640, 1280, 10240),
            (640, 5120, 1280), (1232, 1024, 640),
            (4096, 1024, 640), (1232, 1024, 1280),
            (4096, 1024, 1280)):
        module = torch.nn.Linear(inputs, outputs).cuda().eval()
        value = torch.randn(rows, inputs, device="cuda")
        candidate = make_forward(module)
        baseline = module(value)
        actual = candidate(value)
        # A small FP64 subproblem gives a separate arithmetic reference.
        reference = torch.nn.functional.linear(
            value[:16].double(), module.weight.double(),
            module.bias.double())
        relative_error = float((actual[:16].double() - reference).norm() /
                               reference.norm())
        # TF32x3 is approximate. Report the measured error rather than
        # treating an arbitrary operator tolerance as a video-quality gate.
        assert torch.isfinite(actual).all()
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        record = dict(shape=[rows, inputs, outputs],
                      relative_l2_vs_fp64=relative_error,
                      relative_l2_vs_fp32=float((actual - baseline).norm() /
                                                baseline.norm()))
        if not args.accuracy_only:
            baseline_s = graph_seconds(module, value, 20, 5)
            candidate_s = graph_seconds(candidate, value, 20, 5)
            record.update(baseline_s=baseline_s, candidate_s=candidate_s,
                          speedup=baseline_s / candidate_s)
        print("TF32X3_LINEAR_CHECK " + json.dumps(record), flush=True)
        del module, value, candidate, baseline, actual, reference
        torch.cuda.empty_cache()
    print("TF32X3_LINEAR_CHECK completed=true convolution_wrapping=false "
          "full_video_quality=unverified", flush=True)


if __name__ == "__main__":
    main()
