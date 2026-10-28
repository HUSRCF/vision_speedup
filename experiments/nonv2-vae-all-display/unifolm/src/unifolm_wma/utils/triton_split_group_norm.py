"""Inference-only split-row GroupNorm and SiLU.

Mirror PyTorch 2.3's half mean/rstd and intermediate output rounding. Splitting
large groups exposes more parallelism than one CUDA block per sample/group.
Reduction order differs, so full video quality must be validated separately.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _partial(X, Stats, SIZE: tl.constexpr, PARTS: tl.constexpr,
             GROUPS: tl.constexpr, CHANNELS: tl.constexpr,
             FRAMES: tl.constexpr, HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
             S0: tl.constexpr, S1: tl.constexpr, ST: tl.constexpr,
             SH: tl.constexpr, SW: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    off = part * BLOCK + tl.arange(0, BLOCK)
    valid = off < SIZE
    spatial = FRAMES * HEIGHT * WIDTH
    channel = (row % GROUPS) * (CHANNELS // GROUPS) + off // spatial
    rem = off % spatial
    position = ((row // GROUPS) * S0 + channel * S1 +
                (rem // (HEIGHT * WIDTH)) * ST +
                ((rem // WIDTH) % HEIGHT) * SH + (rem % WIDTH) * SW)
    value = tl.load(X + position, valid, other=0).to(tl.float32)
    count = tl.minimum(BLOCK, SIZE - part * BLOCK)
    mean = tl.sum(value, 0) / count
    centered = tl.where(valid, value - mean, 0.)
    m2 = tl.sum(centered * centered, 0)
    tl.store(Stats + (row * PARTS + part) * 2, mean)
    tl.store(Stats + (row * PARTS + part) * 2 + 1, m2)


@triton.jit
def _finish_stats(Stats, Moments, SIZE: tl.constexpr, PARTS: tl.constexpr,
                  EPS: tl.constexpr, HALF: tl.constexpr,
                  TILE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    valid = part < PARTS
    count = tl.where(valid, tl.minimum(TILE, SIZE - part * TILE), 0)
    mean = tl.load(Stats + (row * PARTS + part) * 2, valid, other=0)
    m2 = tl.load(Stats + (row * PARTS + part) * 2 + 1, valid, other=0)
    overall = tl.sum(mean * count, 0) / SIZE
    delta = mean - overall
    variance = tl.sum(m2 + delta * delta * count, 0) / SIZE
    # Native half GroupNorm quantizes eps and both returned statistics.
    epsilon = tl.full((), EPS, tl.float32)
    if HALF:
        epsilon = epsilon.to(tl.float16).to(tl.float32)
    inverse = tl.math.rsqrt(variance + epsilon)
    tl.store(Moments + row * 2, overall)
    tl.store(Moments + row * 2 + 1, inverse)


@triton.jit
def _apply(X, Weight, Bias, Moments, Output, SIZE: tl.constexpr,
           GROUPS: tl.constexpr, CHANNELS: tl.constexpr,
           FRAMES: tl.constexpr, HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
           S0: tl.constexpr, S1: tl.constexpr, ST: tl.constexpr,
           SH: tl.constexpr, SW: tl.constexpr, HALF: tl.constexpr,
           SILU: tl.constexpr,
           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = off < SIZE
    spatial = FRAMES * HEIGHT * WIDTH
    channel = (row % GROUPS) * (CHANNELS // GROUPS) + off // spatial
    rem = off % spatial
    position = ((row // GROUPS) * S0 + channel * S1 +
                (rem // (HEIGHT * WIDTH)) * ST +
                ((rem // WIDTH) % HEIGHT) * SH + (rem % WIDTH) * SW)
    value = tl.load(X + position, valid, other=0).to(tl.float32)
    mean = tl.load(Moments + row * 2).to(tl.float32)
    inverse = tl.load(Moments + row * 2 + 1).to(tl.float32)
    weight = tl.load(Weight + channel, valid, other=0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0).to(tl.float32)
    scale = inverse * weight
    shift = tl.math.fma(-scale, mean, bias)
    normalized = tl.math.fma(scale, value, shift)
    if HALF:
        normalized = normalized.to(tl.float16).to(tl.float32)
    result = normalized
    if SILU:
        result = normalized / (1. + tl.exp(-normalized))
    tl.store(Output + row * SIZE + off, result, valid)


def split_group_norm_silu(value, norm, *, apply_silu=True, native_forward=None):
    norm_dtype = (value.dtype if type(norm) is torch.nn.GroupNorm else
                  getattr(norm, "_inference_dtype", torch.float32))
    if (not value.is_cuda or value.dtype not in (torch.float16, torch.float32) or
            value.ndim not in (4, 5) or value.numel() == 0 or
            value.shape[1] != norm.num_channels or
            torch.is_grad_enabled() or norm.training or
            norm.weight is None or norm.bias is None or
            norm.weight.dtype != value.dtype or norm.bias.dtype != value.dtype or
            norm.weight.device != value.device or norm.bias.device != value.device or
            not norm.weight.is_contiguous() or not norm.bias.is_contiguous() or
            norm_dtype != value.dtype):
        output = norm(value) if native_forward is None else native_forward(value)
        return F.silu(output) if apply_silu else output
    batch, channels = value.shape[:2]
    if value.ndim == 4:
        frames, height, width = 1, *value.shape[2:]
        s0, s1, sh, sw = value.stride()
        st = 0
    else:
        frames, height, width = value.shape[2:]
        s0, s1, st, sh, sw = value.stride()
    groups = norm.num_groups
    size = channels // groups * frames * height * width
    tile = 4096
    parts = triton.cdiv(size, tile)
    stats = torch.empty((batch * groups, parts, 2), device=value.device,
                        dtype=torch.float32)
    moments = torch.empty((batch * groups, 2), device=value.device,
                          dtype=value.dtype)
    output = torch.empty(value.shape, device=value.device, dtype=value.dtype)
    geometry = (groups, channels, frames, height, width, s0, s1, st, sh, sw)
    _partial[(batch * groups, parts)](
        value, stats, size, parts, *geometry, tile, num_warps=4,
        enable_fp_fusion=False)
    _finish_stats[(batch * groups,)](
        stats, moments, size, parts, norm.eps, value.dtype == torch.float16, tile,
        triton.next_power_of_2(parts), num_warps=4, enable_fp_fusion=False)
    _apply[(batch * groups, triton.cdiv(size, 1024))](
        value, norm.weight, norm.bias, moments, output, size, *geometry,
        value.dtype == torch.float16, apply_silu, 1024, num_warps=4,
        enable_fp_fusion=False)
    return output


def install_split_group_norm(model):
    """Replace eligible eval Sequential prefixes; preserve module state keys."""
    installed = 0
    for sequence in model.modules():
        if (type(sequence) is not torch.nn.Sequential or len(sequence) < 2 or
                not isinstance(sequence[0], torch.nn.GroupNorm) or
                type(sequence[1]) is not torch.nn.SiLU or
                sequence[1].inplace):
            continue
        if getattr(sequence, "_split_group_norm_installed", False):
            raise ValueError("Split GroupNorm already installed")
        original = sequence.forward

        def forward(value, sequence=sequence, original=original):
            if sequence.training or torch.is_grad_enabled():
                return original(value)
            value = split_group_norm_silu(value, sequence[0])
            for layer in tuple(sequence.children())[2:]:
                value = layer(value)
            return value

        sequence.forward = forward
        sequence._split_group_norm_installed = True
        installed += 1
    return installed


def install_split_transformer_norms(model):
    """Parallelize transformer input norms without applying an activation."""
    from unifolm_wma.modules.attention import SpatialTransformer, TemporalTransformer
    installed = 0
    for transformer in model.modules():
        if not isinstance(transformer, (SpatialTransformer, TemporalTransformer)):
            continue
        norm = transformer.norm
        if getattr(norm, "_split_group_norm_installed", False):
            raise ValueError("Split transformer GroupNorm already installed")
        original = norm.forward

        def forward(value, norm=norm, original=original):
            return split_group_norm_silu(value, norm, apply_silu=False,
                                         native_forward=original)

        norm.forward = forward
        norm._split_group_norm_installed = True
        installed += 1
    return installed
