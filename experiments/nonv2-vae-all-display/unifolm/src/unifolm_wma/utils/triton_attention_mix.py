"""Fuse attention branch scaling/additions, preserving FP16 rounding points."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mix(Base, Image, State, Action, Output, N: tl.constexpr,
         HAS_IMAGE: tl.constexpr, HAS_STATE: tl.constexpr,
         HAS_ACTION: tl.constexpr, IMAGE_SCALE: tl.constexpr,
         STATE_SCALE: tl.constexpr, ACTION_SCALE: tl.constexpr, HALF: tl.constexpr,
         BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    value = tl.load(Base + index, valid, other=0).to(tl.float32)
    image = tl.full((BLOCK,), 0, tl.float32)
    state = tl.full((BLOCK,), 0, tl.float32)
    action = tl.full((BLOCK,), 0, tl.float32)
    if HAS_IMAGE:
        image = tl.load(Image + index, valid, other=0).to(tl.float32) * IMAGE_SCALE
    if HAS_STATE:
        state = tl.load(State + index, valid, other=0).to(tl.float32) * STATE_SCALE
    if HAS_ACTION:
        action = tl.load(Action + index, valid, other=0).to(tl.float32) * ACTION_SCALE
    # Keep the same left-to-right FP32 rounding as the eager expression.
    if HALF:
        image = image.to(tl.float16).to(tl.float32)
        state = state.to(tl.float16).to(tl.float32)
        action = action.to(tl.float16).to(tl.float32)
        value = (value + image).to(tl.float16).to(tl.float32)
        value = (value + state).to(tl.float16).to(tl.float32)
        value = value + action
    else:
        value = ((value + image) + state) + action
    tl.store(Output + index, value, valid)


def fused_attention_mix(base, image, state, action, scales):
    tensors = (base, image, state, action)
    if (not base.is_cuda or base.dtype not in (torch.float32, torch.float16) or
            any(value is not None and (
                value.shape != base.shape or value.device != base.device or
                value.dtype != base.dtype or not value.is_contiguous())
                for value in tensors)):
        image = 0.0 if image is None else image
        state = 0.0 if state is None else state
        action = 0.0 if action is None else action
        return base + scales[0] * image + scales[1] * state + scales[2] * action
    result = torch.empty_like(base)
    count = base.numel()
    if count:
        _mix[(triton.cdiv(count, 1024),)](
            base, base if image is None else image,
            base if state is None else state,
            base if action is None else action, result, count,
            image is not None, state is not None, action is not None,
            *scales, HALF=base.dtype == torch.float16, BLOCK=1024,
            enable_fp_fusion=False)
    return result
