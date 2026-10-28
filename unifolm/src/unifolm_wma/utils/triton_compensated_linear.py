"""Experimental TF32 three-product GEMM with short FP32 accumulation blocks.

Inputs and outputs stay FP32. This remains approximate arithmetic and needs
complete video validation before use in a competition result.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _split(value):
    bits = value.to(tl.int32, bitcast=True)
    high_bits = (bits + 0xFFF + ((bits >> 13) & 1)) & -8192
    high = high_bits.to(tl.float32, bitcast=True)
    return high, value - high


@triton.jit
def _rounding_boundary(value):
    # An opaque identity prevents dot/add combining from extending a tensor
    # core accumulator across the entire reduction dimension.
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;", constraints="=f,f", args=[value],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _compensated_gemm(X, Weight, Bias, Output,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr,
                      GROUP_M: tl.constexpr):
    if GROUP_M:
        # Finish all columns for a small group of row tiles before moving
        # to the next group, keeping the working set smaller in L2.
        pid = tl.program_id(0)
        row_tiles = tl.cdiv(M, BM)
        col_tiles = tl.cdiv(N, BN)
        group = pid // (GROUP_M * col_tiles)
        first_row = group * GROUP_M
        group_rows = tl.minimum(GROUP_M, row_tiles - first_row)
        within_group = pid % (GROUP_M * col_tiles)
        row_tile = first_row + within_group % group_rows
        col_tile = within_group // group_rows
    else:
        row_tile = tl.program_id(0)
        col_tile = tl.program_id(1)
    rows = row_tile * BM + tl.arange(0, BM)
    cols = col_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        weight = tl.load(Weight + cols[None, :] * K + k[:, None],
                         (cols[None, :] < N) & (k[:, None] < K), other=0)
        x_high, x_low = _split(x)
        w_high, w_low = _split(weight)
        primary = _rounding_boundary(tl.dot(x_high, w_high, allow_tf32=True))
        correction = tl.dot(x_low, w_high, allow_tf32=True)
        correction = tl.dot(x_high, w_low, correction, allow_tf32=True)
        block_result = _rounding_boundary(primary + correction)
        accumulator += block_result
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    tl.store(Output + rows[:, None] * N + cols[None, :], accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))


def compensated_linear(value, weight, bias=None, *, block_k=32,
                       block_m=32, block_n=64, group_m=0,
                       num_warps=4, num_stages=2, return_kernel=False):
    if (not value.is_cuda or value.dtype != torch.float32 or
            weight.dtype != torch.float32 or value.device != weight.device or
            not value.is_contiguous() or not weight.is_contiguous() or
            value.shape[-1] != weight.shape[1]):
        raise ValueError("Compensated GEMM requires contiguous CUDA FP32 inputs")
    rows = value.numel() // value.shape[-1]
    channels, reduction = weight.shape
    result = torch.empty((*value.shape[:-1], channels), device=value.device,
                         dtype=value.dtype)
    row_tiles = triton.cdiv(rows, block_m)
    col_tiles = triton.cdiv(channels, block_n)
    grid = ((row_tiles * col_tiles,) if group_m else (row_tiles, col_tiles))
    kernel = _compensated_gemm[grid](
        value, weight, weight if bias is None else bias, result,
        rows, channels, reduction, bias is not None,
        BM=block_m, BN=block_n, BK=block_k, GROUP_M=group_m,
        num_warps=num_warps, num_stages=num_stages,
        enable_fp_fusion=False)
    return (result, kernel) if return_kernel else result


def make_forward(module, *, min_matmul_macs=500_000_000, narrow_output=False,
                 narrow_ffn=False, backend="tf32", cache_half_weights=False, group_m=0,
                 block_k=32, pack_inputs=False):
    if narrow_output and narrow_ffn:
        raise ValueError("A Linear cannot use both narrow shape guards")
    if backend not in {"tf32", "scaled-fp16"}:
        raise ValueError("Unknown compensated Linear backend")
    if cache_half_weights and backend != 'scaled-fp16':
        raise ValueError('Packed weights require scaled FP16 backend')
    if group_m not in {0, 4, 8, 16, 32} or (group_m and backend != 'scaled-fp16'):
        raise ValueError('Grouped scheduling requires scaled FP16 backend and a supported group')
    if block_k not in {32, 64, 128} or (block_k != 32 and backend != 'scaled-fp16'):
        raise ValueError('Larger reduction tiles require scaled FP16 backend')
    if pack_inputs and not cache_half_weights:
        raise ValueError('Input packing requires cached compensated Half weights')
    original = module.forward
    packed_key, packed_parts = None, None

    def forward(value):
        nonlocal packed_key, packed_parts
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled() or not value.is_cuda
                or value.dtype != torch.float32
                or module.weight.dtype != torch.float32
                or not value.is_contiguous() or not module.weight.is_contiguous()
                or value.numel() * module.out_features < min_matmul_macs):
            return original(value)
        if narrow_output and (module.out_features != 320 or
                              module.in_features not in (320, 1280) or
                              value.numel() // module.in_features != 40960):
            return original(value)
        if narrow_ffn and (module.in_features != 320 or
                           module.out_features != 2560 or
                           value.numel() // module.in_features != 40960):
            return original(value)
        if backend == "scaled-fp16":
            from .triton_half_compensated_linear import half_compensated_linear, pack_half_weight
            rows = value.numel() // module.in_features
            parts = None
            if cache_half_weights:
                weight = module.weight
                key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.device)
                if packed_key == key:
                    parts = packed_parts
                elif not torch.cuda.is_current_stream_capturing():
                    # Retain storage on updates so existing graph readers see
                    # refreshed parts. Cold captures use the uncached kernel.
                    reuse = (packed_parts if packed_parts is not None and
                             packed_parts[0].shape == weight.shape and
                             packed_parts[0].device == weight.device else None)
                    packed_parts = pack_half_weight(weight, out=reuse)
                    packed_key = key
                    parts = packed_parts
            if pack_inputs and module.out_features >= 1280 and parts is not None:
                from .triton_prepacked_input_linear import prepacked_input_linear
                return prepacked_input_linear(value, module.weight, module.bias,
                    block_m=64 if rows >= 2560 else 32, weight_parts=parts,
                    group_m=group_m, block_k=block_k)
            return half_compensated_linear(value, module.weight, module.bias,
                                            block_m=64 if rows >= 2560 else 32,
                                            weight_parts=parts, group_m=group_m, block_k=block_k)
        return compensated_linear(value, module.weight, module.bias)

    return forward


def install_compensated_linears(model, *, narrow_output=False, narrow_ffn=False,
                                backend="tf32", cache_half_weights=False, group_m=0,
                                block_k=32, pack_inputs=False):
    count = 0
    narrow_count = 0
    ffn_count = 0
    for module in model.modules():
        if (isinstance(module, torch.nn.Linear) and
                (min(module.in_features, module.out_features) >= 640 or
                 (narrow_output and module.out_features == 320 and
                  module.in_features in (320, 1280)) or
                 (narrow_ffn and module.in_features == 320 and
                  module.out_features == 2560))):
            if getattr(module, "_tf32x3_installed", False):
                raise RuntimeError("Compensated Linear already installed")
            is_narrow = min(module.in_features, module.out_features) < 640
            is_ffn = is_narrow and module.out_features == 2560
            module.forward = make_forward(module, narrow_output=is_narrow and not is_ffn,
                                          narrow_ffn=is_ffn, backend=backend,
                                          cache_half_weights=cache_half_weights, group_m=group_m,
                                          block_k=block_k, pack_inputs=pack_inputs)
            module._tf32x3_installed = True
            count += 1
            narrow_count += is_narrow and not is_ffn
            ffn_count += is_ffn
    counts = {"Linear": count}
    if narrow_output:
        counts['NarrowOutputLinear'] = narrow_count
    if narrow_ffn:
        counts['NarrowFFNLinear'] = ffn_count
    return counts
