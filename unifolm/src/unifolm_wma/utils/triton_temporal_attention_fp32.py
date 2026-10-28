"""IEEE FP32 16-frame attention without intermediate global tensors."""
import torch; from .attention_buffer_allocation import empty_like as empty_attention_output
import triton
import triton.language as tl


@triton.jit
def _attention_fp32(Q, K, V, Output, QK, Probability,
                    HEADS: tl.constexpr, SCALE: tl.constexpr,
                    DEBUG: tl.constexpr):
    batch_head = tl.program_id(0)
    batch, head = batch_head // HEADS, batch_head % HEADS
    frame = tl.arange(0, 16)
    channel = tl.arange(0, 64)
    origin = batch * 16 * HEADS * 64 + head * 64
    q = tl.load(Q + origin + frame[:, None] * HEADS * 64 + channel[None, :])
    k = tl.load(K + origin + frame[None, :] * HEADS * 64 + channel[:, None])
    scores = tl.dot(q, k, allow_tf32=False)
    debug_offset = batch_head * 256 + frame[:, None] * 16 + frame[None, :]
    if DEBUG:
        tl.store(QK + debug_offset, scores)
    scores = scores * SCALE
    probabilities = tl.math.exp(scores - tl.max(scores, 1)[:, None])
    total = tl.sum(tl.reshape(probabilities, (16, 2, 8)), 1)
    total = tl.sum(tl.reshape(total, (16, 2, 4)), 1)
    total = tl.sum(tl.reshape(total, (16, 2, 2)), 1)
    denominator = tl.sum(total, 1)[:, None]
    probabilities = tl.math.div_rn(probabilities, denominator)
    if DEBUG:
        tl.store(Probability + debug_offset, probabilities)
    v = tl.load(V + origin + frame[:, None] * HEADS * 64 + channel[None, :])
    result = tl.dot(probabilities, v, allow_tf32=False)
    tl.store(Output + origin + frame[:, None] * HEADS * 64 + channel[None, :], result)


def fused_temporal_attention_fp32(query, key, value, heads, scale,
                                  return_intermediates=False):
    if (query.ndim != 3 or query.shape[1:] != (16, heads * 64) or
            query.shape != key.shape or query.shape != value.shape or
            any(not t.is_cuda or t.dtype != torch.float32 or
                not t.is_contiguous() or t.device != query.device
                for t in (query, key, value)) or torch.is_grad_enabled() or
            torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('Requires contiguous IEEE FP32 B,16,H*64 inference')
    output = empty_attention_output('temporal', query)
    if return_intermediates:
        qk = torch.empty((query.shape[0] * heads, 16, 16),
                         device=query.device, dtype=query.dtype)
        probabilities = torch.empty_like(qk)
    else:
        qk = probabilities = output
    if query.shape[0]:
        _attention_fp32[(query.shape[0] * heads,)](
            query, key, value, output, qk, probabilities, heads, scale,
            return_intermediates, num_warps=4, enable_fp_fusion=False)
    return (output, qk, probabilities) if return_intermediates else output
