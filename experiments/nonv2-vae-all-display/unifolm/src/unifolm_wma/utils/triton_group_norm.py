"""Inference-only fused GroupNorm variants for contiguous CUDA tensors."""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # Keep the default inference path usable without Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _group_norm_kernel(x_ptr, weight_ptr, bias_ptr, scale_ptr, shift_ptr,
                           output_ptr, scale_stride_n: tl.constexpr,
                           scale_stride_c: tl.constexpr,
                           shift_stride_n: tl.constexpr,
                           shift_stride_c: tl.constexpr,
                           channels: tl.constexpr, groups: tl.constexpr,
                           spatial: tl.constexpr, group_size: tl.constexpr,
                           eps: tl.constexpr, has_modulation: tl.constexpr,
                           apply_silu: tl.constexpr,
                           block_size: tl.constexpr):
        row = tl.program_id(0)
        batch = row // groups
        group = row - batch * groups
        offsets = tl.arange(0, block_size)
        mask = offsets < group_size
        base = row * group_size
        values = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        values = values.to(tl.float32)
        mean = tl.sum(tl.where(mask, values, 0.0), axis=0) / group_size
        centered = tl.where(mask, values - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / group_size
        normalized = centered / tl.sqrt(variance + eps)

        channels_per_group = channels // groups
        channel = group * channels_per_group + offsets // spatial
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)
        output = normalized * weight + bias
        if has_modulation:
            scale = tl.load(
                scale_ptr + batch * scale_stride_n + channel * scale_stride_c,
                mask=mask, other=0.0)
            shift = tl.load(
                shift_ptr + batch * shift_stride_n + channel * shift_stride_c,
                mask=mask, other=0.0)
            output = output * (1.0 + scale) + shift
        if apply_silu:
            output = output * tl.sigmoid(output)
        tl.store(output_ptr + base + offsets, output, mask=mask)


    @triton.jit
    def _framewise_group_norm_kernel(
            x_ptr, weight_ptr, bias_ptr, output_ptr,
            channels: tl.constexpr, frames: tl.constexpr,
            height: tl.constexpr, width: tl.constexpr,
            x_stride_b: tl.constexpr, x_stride_c: tl.constexpr,
            x_stride_t: tl.constexpr, x_stride_h: tl.constexpr,
            x_stride_w: tl.constexpr, groups: tl.constexpr,
            group_size: tl.constexpr, eps: tl.constexpr,
            block_size: tl.constexpr):
        row = tl.program_id(0)
        group = row % groups
        frame_row = row // groups
        frame = frame_row % frames
        batch = frame_row // frames
        offsets = tl.arange(0, block_size)
        mask = offsets < group_size
        channels_per_group = channels // groups
        spatial = height * width
        channel_in_group = offsets // spatial
        spatial_offset = offsets - channel_in_group * spatial
        y = spatial_offset // width
        x = spatial_offset - y * width
        channel = group * channels_per_group + channel_in_group
        source_offset = (batch * x_stride_b + channel * x_stride_c +
                         frame * x_stride_t + y * x_stride_h +
                         x * x_stride_w)
        values = tl.load(x_ptr + source_offset, mask=mask, other=0.0)
        values = values.to(tl.float32)
        mean = tl.sum(tl.where(mask, values, 0.0), axis=0) / group_size
        centered = tl.where(mask, values - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / group_size
        normalized = centered / tl.sqrt(variance + eps)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)
        output = normalized * weight + bias
        output = output * tl.sigmoid(output)
        output_offset = (((batch * channels + channel) * frames + frame) *
                         spatial + spatial_offset)
        tl.store(output_ptr + output_offset, output, mask=mask)


    @triton.jit
    def _temporal_group_norm_kernel(
            x_ptr, weight_ptr, bias_ptr, output_ptr,
            channels: tl.constexpr, frames: tl.constexpr,
            height: tl.constexpr, width: tl.constexpr,
            x_stride_b: tl.constexpr, x_stride_c: tl.constexpr,
            x_stride_t: tl.constexpr, x_stride_h: tl.constexpr,
            x_stride_w: tl.constexpr, groups: tl.constexpr,
            group_size: tl.constexpr, eps: tl.constexpr,
            block_size: tl.constexpr, chunks: tl.constexpr):
        row = tl.program_id(0)
        batch = row // groups
        group = row - batch * groups
        lanes = tl.arange(0, block_size)
        channels_per_group = channels // groups
        frame_spatial = frames * height * width
        spatial = height * width
        value_sum = 0.0
        square_sum = 0.0
        for chunk in tl.static_range(0, chunks):
            offsets = chunk * block_size + lanes
            mask = offsets < group_size
            channel_in_group = offsets // frame_spatial
            remainder = offsets - channel_in_group * frame_spatial
            frame = remainder // spatial
            spatial_offset = remainder - frame * spatial
            y = spatial_offset // width
            x = spatial_offset - y * width
            channel = group * channels_per_group + channel_in_group
            source_offset = (batch * x_stride_b + channel * x_stride_c +
                             frame * x_stride_t + y * x_stride_h +
                             x * x_stride_w)
            values = tl.load(x_ptr + source_offset, mask=mask, other=0.0)
            values = values.to(tl.float32)
            value_sum += tl.sum(tl.where(mask, values, 0.0), axis=0)
            square_sum += tl.sum(tl.where(mask, values * values, 0.0), axis=0)
        mean = value_sum / group_size
        variance = square_sum / group_size - mean * mean
        inverse_std = 1.0 / tl.sqrt(variance + eps)
        for chunk in tl.static_range(0, chunks):
            offsets = chunk * block_size + lanes
            mask = offsets < group_size
            channel_in_group = offsets // frame_spatial
            remainder = offsets - channel_in_group * frame_spatial
            frame = remainder // spatial
            spatial_offset = remainder - frame * spatial
            y = spatial_offset // width
            x = spatial_offset - y * width
            channel = group * channels_per_group + channel_in_group
            source_offset = (batch * x_stride_b + channel * x_stride_c +
                             frame * x_stride_t + y * x_stride_h +
                             x * x_stride_w)
            values = tl.load(x_ptr + source_offset, mask=mask, other=0.0)
            values = values.to(tl.float32)
            weight = tl.load(weight_ptr + channel, mask=mask, other=0.0)
            bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)
            output = (values - mean) * inverse_std * weight + bias
            output = output * tl.sigmoid(output)
            output_offset = (((batch * channels + channel) * frames + frame) *
                             spatial + spatial_offset)
            tl.store(output_ptr + output_offset, output, mask=mask)


def fused_group_norm(x, norm, *, scale=None, shift=None, apply_silu=True):
    """Apply GroupNorm and optional FiLM modulation/SiLU in one CUDA kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for fused GroupNorm")
    if not x.is_cuda or x.ndim != 4:
        raise ValueError("fused GroupNorm requires a 4D CUDA tensor")
    if x.dtype != torch.float32:
        raise ValueError("fused GroupNorm currently requires float32 input")
    if norm.weight is None or norm.bias is None:
        raise ValueError("fused GroupNorm requires affine weight and bias")
    has_modulation = scale is not None or shift is not None
    if has_modulation and (scale is None or shift is None):
        raise ValueError("scale and shift must be provided together")

    def native_fallback():
        output = norm(x)
        if has_modulation:
            output = output * (1.0 + scale) + shift
        return F.silu(output) if apply_silu else output

    if not x.is_contiguous():
        return native_fallback()

    batch, channels = x.shape[:2]
    groups = norm.num_groups
    spatial = x.numel() // (batch * channels)
    group_size = (channels // groups) * spatial
    block_size = triton.next_power_of_2(group_size)
    if block_size > 65536:
        return native_fallback()
    output = torch.empty_like(x)
    if not has_modulation:
        scale = x
        shift = x
        scale_stride_n = scale_stride_c = 0
        shift_stride_n = shift_stride_c = 0
    else:
        scale_stride_n, scale_stride_c = scale.stride()[:2]
        shift_stride_n, shift_stride_c = shift.stride()[:2]
    num_warps = 8 if block_size >= 8192 else 4
    _group_norm_kernel[(batch * groups,)](
        x, norm.weight, norm.bias, scale, shift, output,
        scale_stride_n, scale_stride_c, shift_stride_n, shift_stride_c,
        channels, groups, spatial, group_size, norm.eps, has_modulation,
        apply_silu, block_size, num_warps=num_warps)
    return output


def fused_framewise_group_norm_silu(x, norm):
    """Approximate 5D GroupNorm by normalizing each video frame separately."""
    if triton is None:
        raise RuntimeError("Triton is required for fused GroupNorm")
    if not x.is_cuda or x.ndim != 5:
        raise ValueError(
            "framewise fused GroupNorm requires a 5D CUDA tensor")
    if x.dtype != torch.float32:
        raise ValueError("framewise fused GroupNorm currently requires float32")
    batch, channels, frames, height, width = x.shape
    spatial = height * width
    groups = norm.num_groups
    group_size = (channels // groups) * spatial
    block_size = triton.next_power_of_2(group_size)
    if block_size > 65536:
        raise ValueError(
            f"Framewise GroupNorm row is too large for fused kernel: {group_size}")
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    num_warps = 8 if block_size >= 8192 else 4
    _framewise_group_norm_kernel[(batch * frames * groups,)](
        x, norm.weight, norm.bias, output, channels, frames, height, width,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        groups, group_size, norm.eps, block_size, num_warps=num_warps)
    return output


def fused_temporal_group_norm_silu(x, norm):
    """Apply exact 5D GroupNorm and SiLU in one persistent Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for fused GroupNorm")
    if not x.is_cuda or x.ndim != 5:
        raise ValueError("fused temporal GroupNorm requires a 5D CUDA tensor")
    if x.dtype != torch.float32:
        raise ValueError("fused temporal GroupNorm currently requires float32")
    batch, channels, frames, height, width = x.shape
    groups = norm.num_groups
    group_size = (channels // groups) * frames * height * width
    block_size = min(32768, triton.next_power_of_2(group_size))
    chunks = triton.cdiv(group_size, block_size)
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _temporal_group_norm_kernel[(batch * groups,)](
        x, norm.weight, norm.bias, output, channels, frames, height, width,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        groups, group_size, norm.eps, block_size, chunks, num_warps=8)
    return output
