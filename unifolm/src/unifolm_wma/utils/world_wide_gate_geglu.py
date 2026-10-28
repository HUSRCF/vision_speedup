"""World wide GEGLU: protect value branch and accelerate only the gate."""
import torch
from .triton_zero_weight_low_geglu import residual_count as refresh_geglu_count
from .triton_half_compensated_linear import pack_half_weight
from .triton_branch_protected_geglu import branch_protected_geglu


def make_forward(module, mode):
    if mode not in {'gate'}:
        raise ValueError('Unknown world GEGLU mode')
    original = module.forward
    projection = module.proj
    signature, parts = None, None
    selected = {(10240, 640), (2560, 1280), (640, 1280)}

    def forward(value):
        nonlocal signature, parts
        if (module.training or projection.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32 or not value.is_cuda
                or value.dtype != torch.float32 or value.ndim < 1
                or not value.is_contiguous()
                or value.shape[-1] != projection.in_features
                or (value.numel() // projection.in_features,
                    projection.in_features) not in selected
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
            refresh_geglu_count(parts,refresh=True)
            signature = key
        return branch_protected_geglu(value, parts, projection.bias, mode=mode,
            block_n=64 if projection.in_features==640 else 32, group_m=32)

    return forward


def install_world_wide_gate_geglus(diffusion, mode, scope='all'):
    from unifolm_wma.modules.attention import GEGLU
    if mode not in {'gate'} or scope not in {'all', '640', '1280'}:
        raise ValueError('Invalid world primary GEGLU configuration')
    count = 0
    for module in diffusion.modules():
        if (isinstance(module, GEGLU) and module.proj.in_features in {640, 1280}
                and module.proj.out_features == 8 * module.proj.in_features
                and (scope == 'all' or module.proj.in_features == int(scope))):
            if (not getattr(module, '_decision_primary_geglu_installed', False)
                    or not getattr(module, '_prepacked_geglu_installed', False)
                    or getattr(module, '_world_wide_gate_geglu_installed', False)):
                raise ValueError('World GEGLU requires exactly one accepted wide wrapper')
            module.forward = make_forward(module, mode)
            module._world_wide_gate_geglu_installed = True
            count += 1
    return count
