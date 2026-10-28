"""Wide GEGLU branch protection, using the accepted BK64 kernel arithmetic."""
import torch; from .triton_zero_weight_low_geglu import run as run_zero_geglu, residual_count as refresh_geglu_count
from .geglu_output_allocation import empty as empty_geglu_output
import triton
from .triton_half_compensated_linear import pack_half_weight
from .triton_world_narrow_geglu import _geglu


def branch_protected_geglu(value, parts, bias=None, *, mode='value', block_n=32,
                         debug_branches=False, debug_poison=False, block_k=64, group_m=32):
    if (mode not in {'compensated', 'value', 'gate', 'both'} or block_n not in {32, 64}
            or block_k != 64 or group_m not in {16, 32}):
        raise ValueError('Unsupported narrow GEGLU experiment')
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim < 1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous inference CUDA FP32 input')
    if len(parts) != 2 or any(p.ndim != 2 or p.shape != parts[0].shape
            or p.shape[1] != value.shape[-1] or p.shape[0] % 2
            or p.dtype != torch.float16 or p.device != value.device
            or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid GEGLU half weight parts')
    rows = value.numel() // value.shape[-1]
    channels, inner = parts[0].shape[0] // 2, value.shape[-1]
    if min(rows, channels, inner) <= 0:
        raise ValueError('Empty shapes unsupported')
    if bias is not None and (bias.shape != (channels * 2,) or bias.dtype != torch.float32
            or bias.device != value.device or not bias.is_contiguous()):
        raise ValueError('Invalid GEGLU FP32 bias')
    inputs = pack_half_weight(value.view(rows, inner), weight_scale=1.)
    output = empty_geglu_output((*value.shape[:-1], channels), device=value.device, dtype=torch.float32)
    v = torch.empty_like(output) if debug_branches else output
    g = torch.empty_like(output) if debug_branches else output
    if debug_poison:
        output.fill_(float('nan'))
    if not run_zero_geglu(inputs,parts,bias,output,v,g,rows,channels,inner,64,block_n,group_m,mode in {"value","both"},mode in {"gate","both"},debug_branches,block_k,4,2):
        _geglu[(triton.cdiv(rows, 64) * triton.cdiv(channels, block_n),)](
            *inputs, *parts, parts[0] if bias is None else bias, output, v, g,
            rows, channels, inner, bias is not None, block_n, group_m,
            mode in {'value', 'both'}, mode in {'gate', 'both'}, debug_branches, block_k,
            num_warps=4, num_stages=2, enable_fp_fusion=False)
    return (output, v, g) if debug_branches else output
