"""Cover remaining real shapes before considering a wider FP32-output backend."""
import torch
from benchmark_half_compensated_linear import compare


@torch.no_grad()
def main():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for rows, channels, outputs in ((10240, 2560, 640), (10240, 640, 640),
                                   (2560, 1280, 1280), (640, 1280, 1280),
                                   (640, 1280, 10240), (40960, 320, 320)):
        value = torch.randn(rows, channels, device='cuda')
        weight = torch.randn(outputs, channels, device='cuda') * 0.02
        bias = torch.randn(outputs, device='cuda')
        compare(value, weight, bias, 'remaining-real-shape')
        del value, weight, bias
        torch.cuda.empty_cache()
    print('HALF_COMPENSATED_REMAINING_CHECK passed=true inference_installed=false full_video_quality=unverified', flush=True)


if __name__ == '__main__': main()
