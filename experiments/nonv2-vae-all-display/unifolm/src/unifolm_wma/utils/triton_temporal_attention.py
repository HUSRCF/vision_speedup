"""Fuse 16-frame FP16 attention while retaining materialization rounding."""
import torch
import triton
import triton.language as tl


@triton.jit
def _temporal_attention(Q, K, V, Output, QKDebug, ProbabilityDebug,
                         HEADS: tl.constexpr, SCALE: tl.constexpr,
                         ACCURATE_EXP: tl.constexpr, DEBUG: tl.constexpr):
    batch_head = tl.program_id(0)
    batch, head = batch_head // HEADS, batch_head % HEADS
    frame = tl.arange(0, 16)
    channel = tl.arange(0, 64)
    origin = batch * 16 * HEADS * 64 + head * 64
    q = tl.load(Q + origin + frame[:, None] * HEADS * 64 + channel[None, :])
    k = tl.load(K + origin + frame[None, :] * HEADS * 64 + channel[:, None])
    dot_half = tl.dot(q, k).to(tl.float16)
    scores = dot_half.to(tl.float32)
    debug_offset = batch_head * 256 + frame[:, None] * 16 + frame[None, :]
    if DEBUG:
        tl.store(QKDebug + debug_offset, dot_half)
    # Eager scores are a Half tensor, then scaled into another Half tensor.
    scores = (scores * SCALE).to(tl.float16).to(tl.float32)
    scores = scores - tl.max(scores, 1)[:, None]
    if ACCURATE_EXP:
        probabilities = tl.math.exp(scores)
    else:
        probabilities = tl.exp(scores)
    if ACCURATE_EXP:
        # PyTorch's 16-lane persistent softmax reduces XOR partners in
        # descending distance: 8,4,2,1. Reduce pairs explicitly in that order.
        total = tl.sum(tl.reshape(probabilities, (16,2,8)), 1)
        total = tl.sum(tl.reshape(total, (16,2,4)), 1)
        total = tl.sum(tl.reshape(total, (16,2,2)), 1)
        denominator = tl.sum(total, 1)[:, None]
        probabilities = tl.math.div_rn(probabilities, denominator)
    else:
        denominator = tl.sum(probabilities, 1)[:, None]
        probabilities = probabilities / denominator
    probabilities = probabilities.to(tl.float16)
    if DEBUG:
        tl.store(ProbabilityDebug + debug_offset, probabilities)
    v = tl.load(V + origin + frame[:, None] * HEADS * 64 + channel[None, :])
    result = tl.dot(probabilities, v)
    tl.store(Output + origin + frame[:, None] * HEADS * 64 + channel[None, :], result)


def fused_temporal_attention(query, key, value, heads, scale,
                              accurate_exp=True, return_intermediates=False):
    if (query.ndim != 3 or query.shape[1:] != (16, heads * 64) or
            query.shape != key.shape or query.shape != value.shape or
            any(not tensor.is_cuda or tensor.dtype != torch.float16 or
                not tensor.is_contiguous() or tensor.device != query.device
                for tensor in (query, key, value)) or torch.is_grad_enabled()):
        raise ValueError('Fused temporal attention requires contiguous FP16 B,16,H*64')
    output = torch.empty_like(query)
    if return_intermediates:
        qk = torch.empty((query.shape[0]*heads,16,16),
                         device=query.device,dtype=query.dtype)
        probabilities = torch.empty_like(qk)
    else:
        qk = probabilities = output
    if query.shape[0]:
        _temporal_attention[(query.shape[0] * heads,)](
            query, key, value, output, qk, probabilities, heads, scale,
            accurate_exp, return_intermediates, num_warps=4,
            enable_fp_fusion=False)
    return (output,qk,probabilities) if return_intermediates else output
