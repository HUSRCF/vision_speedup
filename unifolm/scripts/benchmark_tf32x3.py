"""Diagnostic operator accuracy/speed check; not a competition benchmark."""
import json
import statistics
import time

import torch

from unifolm_wma.utils.tf32x3 import make_forward, tensor_core_math


def elapsed(function, x):
    function(x)
    measurements = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function(x)
        torch.cuda.synchronize()
        measurements.append(time.perf_counter() - start)
    return statistics.median(measurements)


@torch.no_grad()
def main():
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    # Small FP64 references quantify arithmetic error independently of FP32.
    for module, shape in (
        (torch.nn.Linear(320, 640), (256, 320)),
        (torch.nn.Conv2d(64, 64, 3, padding=1), (2, 64, 16, 16)),
        (torch.nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0)), (1, 64, 8, 8, 8)),
    ):
        module = module.cuda().eval()
        x = torch.randn(*shape, device="cuda")
        reference = module.double()(x.double())
        module.float()
        fp32 = module(x)
        compensated = make_forward(module, min_matmul_macs=0)(x)
        if isinstance(module, torch.nn.Linear):
            torch.testing.assert_close(make_forward(module)(x), fp32, rtol=0, atol=0)
        with tensor_core_math():
            tf32 = module(x)
        stats = {}
        for label, actual in (("fp32", fp32), ("tf32", tf32), ("tf32x3", compensated)):
            error = actual.double() - reference
            stats[label] = {"relative_l2": (error.norm() / reference.norm()).item(),
                            "max_absolute": error.abs().max().item()}
        print("OPERATOR_ACCURACY", type(module).__name__, json.dumps(stats), flush=True)
        assert stats["tf32x3"]["relative_l2"] < 2e-6, stats
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32

    # Shapes from the actual warmed-up FP32 denoiser profile.
    for module, shape in (
        (torch.nn.Linear(320, 2560), (40960, 320)),
        (torch.nn.Linear(1280, 10240), (2560, 1280)),
        (torch.nn.Conv2d(1280, 1280, 3, padding=1), (16, 1280, 5, 8)),
        (torch.nn.Conv3d(320, 320, (3, 1, 1), padding=(1, 0, 0)), (1, 320, 16, 40, 64)),
    ):
        module = module.cuda().eval()
        x = torch.randn(*shape, device="cuda")
        compensated = make_forward(module)
        fp32_s = elapsed(module, x)
        compensated_s = elapsed(compensated, x)
        with tensor_core_math():
            tf32_s = elapsed(module, x)
        print("OPERATOR_SPEED", json.dumps({"type": type(module).__name__,
              "input_shape": shape, "fp32_s": fp32_s, "tf32_s": tf32_s,
              "tf32x3_s": compensated_s, "speedup": fp32_s / compensated_s}), flush=True)


if __name__ == "__main__":
    main()
