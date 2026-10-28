"""Narrow world upsample cuDNN algorithm-selection experiment, FP32 unchanged."""
import torch


def make_forward(module, channels):
    original = module.forward

    def forward(value):
        spatial = (20, 32) if channels == 1280 else (40, 64)
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype != torch.float32
                or tuple(value.shape) != (16, channels, *spatial)
                or not value.is_contiguous()
                or module.weight.dtype != torch.float32 or module.weight.device != value.device
                or not module.weight.is_contiguous()
                or (module.bias is not None and (module.bias.dtype != torch.float32
                    or module.bias.device != value.device or not module.bias.is_contiguous()))):
            return original(value)
        previous = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.benchmark = False
            result = original(value)
            module._world_upsample_heuristic_calls = getattr(module, '_world_upsample_heuristic_calls', 0)+1
            return result
        finally:
            torch.backends.cudnn.benchmark = previous
    return forward


def install_world_upsample_heuristic(diffusion):
    from unifolm_wma.modules.networks.wma_model import Upsample
    targets = {'output_blocks.5.3':1280, 'output_blocks.8.3':640}
    selected = [(name, module.conv, module.channels) for name, module in diffusion.named_modules()
                if name in targets and type(module) is Upsample and module.use_conv
                and module.channels == targets[name]]
    assert len(selected) == 2 and {c for _,_,c in selected} == {640,1280}
    for name, module, channels in selected:
        assert type(module) is torch.nn.Conv2d
        assert (module.in_channels,module.out_channels,module.kernel_size,module.stride,module.padding,module.dilation,module.groups) == (channels,channels,(3,3),(1,1),(1,1),(1,1),1)
        assert not getattr(module, '_world_upsample_heuristic_installed', False), name
    for name, module, channels in selected:
        module.forward = make_forward(module, channels)
        module._world_upsample_heuristic_installed = True
    return [name+'.conv' for name,_,_ in selected]
