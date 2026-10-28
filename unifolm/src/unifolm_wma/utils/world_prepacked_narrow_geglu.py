"""World C320 three-product GEGLU with static weight parts and dynamic inputs."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_prepacked_geglu import prepacked_geglu


def checked_tile(tile):
    bm, bn, group = tile
    if bm not in {32, 64} or bn not in {32, 64} or group not in {16, 32}:
        raise ValueError('Unsupported world narrow GEGLU tile')
    return bm, bn, group


def make_forward(module, tile):
    bm, bn, group = checked_tile(tile)
    original, projection = module.forward, module.proj
    signature, parts = None, None

    def forward(value):
        nonlocal signature, parts
        if (module.training or projection.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32 or not value.is_cuda
                or value.dtype != torch.float32 or value.ndim < 1
                or not value.is_contiguous() or value.shape[-1] != 320
                or value.numel() // 320 != 40960
                or projection.weight.dtype != torch.float32
                or projection.weight.device != value.device
                or not projection.weight.is_contiguous()
                or (projection.bias is not None and
                    (projection.bias.dtype != torch.float32
                     or projection.bias.device != value.device
                     or not projection.bias.is_contiguous()))):
            return original(value)
        weight = projection.weight
        try:
            version = weight._version
        except RuntimeError:
            return original(value)
        key = weight.data_ptr(), version, tuple(weight.shape), weight.dtype, weight.device
        if key != signature:
            if torch.cuda.is_current_stream_capturing():
                return original(value)
            reuse = (parts if parts is not None and parts[0].shape == weight.shape
                     and parts[0].device == weight.device else None)
            parts = pack_half_weight(weight, out=reuse)
            signature = key
        return prepacked_geglu(value, parts, projection.bias, block_m=bm,
            block_n=bn, group_m=group, num_warps=4, num_stages=2)

    return forward


def install_world_prepacked_narrow_geglus(diffusion, tile):
    from unifolm_wma.modules.attention import GEGLU
    checked_tile(tile)
    count = 0
    for module in diffusion.modules():
        if (isinstance(module, GEGLU) and module.proj.in_features == 320
                and module.proj.out_features == 2560):
            if (not getattr(module, '_decision_half_geglu_installed', False)
                    or getattr(module, '_world_prepacked_narrow_geglu_installed', False)):
                raise ValueError('World narrow GEGLU requires accepted decision wrapper')
            module.forward = make_forward(module, tile)
            module._world_prepacked_narrow_geglu_installed = True
            count += 1
    return count
