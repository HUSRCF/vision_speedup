"""Shared device count plus the original short-BK64 FFN and FP32 residual."""
import torch; import os
import triton
import triton.language as tl
from .triton_zero_weight_low_linear import _short_sum
from .gemm_output_allocation import empty as empty_gemm

@triton.jit
def _zero_low_gemm_residual(XH, XL, WH, WL, LowCount, Bias, Residual, Y,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   HAS_BIAS: tl.constexpr, BM: tl.constexpr, GROUP: tl.constexpr,
                   OUTPUT_GROUPS: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(N, 64)
    group = pid // (GROUP * col_tiles)
    first = group * GROUP
    count = tl.minimum(GROUP, row_tiles - first)
    local = pid % (GROUP * col_tiles)
    rows = (first + local % count) * BM + tl.arange(0, BM)
    cols = (local // count) * 64 + tl.arange(0, 64)
    # Uniform dispatch once per tile; both inner loops have static MMA counts.
    if tl.load(LowCount) != 0:
        accumulator = _short_sum(XH, XL, WH, WL, rows, cols, M, N, K, BM, True)
    else:
        accumulator = _short_sum(XH, XL, WH, WL, rows, cols, M, N, K, BM, False)
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    if OUTPUT_GROUPS == 1:
        offset = rows[:, None] * N + cols[None, :]
    else:
        width = N // OUTPUT_GROUPS
        offset = ((cols[None, :] // width) * M * width +
                  rows[:, None] * width + cols[None, :] % width)
    accumulator += tl.load(Residual + rows[:, None] * N + cols[None, :],
                           (rows[:, None] < M) & (cols[None, :] < N), other=0)
    tl.store(Y + offset, accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))



def shared_final_low_count(parts,*,refresh=False):
    """Refresh only static weight metadata; preserve captured count storage."""
    try:
        signature=tuple((p.data_ptr(),p._version,tuple(p.shape),p.dtype,p.device) for p in parts)
    except RuntimeError:
        return None
    cached=getattr(parts[0],'_ffn1280_zero_low_count',None)
    if not refresh and cached is not None and cached[0]==signature:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        return None
    count=torch.count_nonzero(parts[1])
    if cached is not None and cached[1].device==count.device:
        cached[1].copy_(count);count=cached[1]
    parts[0]._ffn1280_zero_low_count=(signature,count)
    return count


def zero_ffn1280_hidden_residual(inputs,parts,count,residual,bias=None):
    high,low=inputs
    if (torch.is_grad_enabled() or not high.is_cuda or high.ndim!=2 or high.shape[1]!=5120
            or min(high.shape)<=0 or any(p.shape!=high.shape or p.device!=high.device
                or p.dtype!=torch.float16 or not p.is_contiguous() for p in inputs)):
        raise ValueError('Expected fresh C1280 hidden Half parts')
    rows,inner=high.shape;channels=parts[0].shape[0]
    if len(parts)!=2 or any(p.shape!=(channels,inner) or p.device!=high.device
            or p.dtype!=torch.float16 or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid original FFN static parts')
    if count.shape!=() or count.dtype!=torch.int64 or count.device!=high.device:
        raise ValueError('Invalid device FFN residual count')
    if (residual.shape!=(rows,channels) or residual.dtype!=torch.float32
            or residual.device!=high.device or not residual.is_contiguous()):
        raise ValueError('Invalid FP32 residual')
    if bias is not None and (bias.shape!=(channels,) or bias.dtype!=torch.float32
            or bias.device!=high.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    output=empty_gemm('classic',(rows,channels),device=high.device,dtype=torch.float32)
    bm=64 if rows>=2560 else 32
    mode=os.environ.get('UNIFOLM_WORLD_FFN1280_ZERO_STAGES','2')
    if mode not in {'2','3'}:raise ValueError('Invalid world C1280 FFN depth')
    stages=3 if mode=='3' and (rows,channels,inner) in {(2560,1280,5120),(640,1280,5120)} else 2
    _zero_low_gemm_residual[(triton.cdiv(rows,bm)*triton.cdiv(channels,64),)](
        *inputs,*parts,count,parts[0] if bias is None else bias,residual,output,
        rows,channels,inner,bias is not None,bm,32,1,
        num_warps=4,num_stages=stages,enable_fp_fusion=False)
    return output
