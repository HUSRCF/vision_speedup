import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split


@triton.jit
def _residual_norm_half_pack(Left, Right, Weight, Bias, Residual, Normalized, High, Low,
                         CHANNELS: tl.constexpr, EPS: tl.constexpr,
                         BLOCK: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < CHANNELS
    offset = row * CHANNELS + channel
    left = tl.load(Left + offset, valid, other=0).to(tl.float32)
    right = tl.load(Right + offset, valid, other=0).to(tl.float32)
    # Preserve the rounding boundary of the separately materialized sum.
    residual = (left + right).to(Left.dtype.element_ty)
    value = residual.to(tl.float32)
    mean = tl.sum(value, 0) / CHANNELS
    centered = tl.where(valid, value - mean, 0.)
    variance = tl.sum(centered * centered, 0) / CHANNELS
    inverse = tl.math.rsqrt(variance + EPS)
    weight = tl.load(Weight + channel, valid, other=0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0).to(tl.float32)
    normalized = ((value - mean) * inverse) * weight + bias
    tl.store(Residual + offset, residual, valid)
    if DEBUG:
        tl.store(Normalized + offset, normalized, valid)
    high, low = _half_split(normalized, 1.)
    # Hide only Half-store address vectorization from AxisInfo. FP32 loads
    # and the residual store still determine the original four-element
    # reduction layout. This identity leaves every address unchanged.
    pack_offset = tl.inline_asm_elementwise("mov.b32 $0, $1;", constraints="=r,r",
        args=[offset], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(High + pack_offset, high, valid)
    tl.store(Low + pack_offset, low, valid)

