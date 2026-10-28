import torch
import triton
import triton.language as tl
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary
from unifolm_wma.utils.triton_half_compensated_linear import _half_split

@triton.jit
def _out_projection_residual(X, High, Low, Bias, Output, Residual, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr, INPUT_PROTECTED: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, ST: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr, TEMPORAL: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles = tl.cdiv(M, BM)
    col_tiles = tl.cdiv(N, BN)
    group = pid // (GROUP_M * col_tiles)
    first_row = group * GROUP_M
    group_rows = tl.minimum(GROUP_M, row_tiles - first_row)
    within_group = pid % (GROUP_M * col_tiles)
    rows = (first_row + within_group % group_rows) * BM + tl.arange(0, BM)
    cols = within_group // group_rows * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        x = tl.load(X + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), other=0)
        if INPUT_PROTECTED:
            x_high, x_low = _half_split(x, 1.0)
        else:
            x_high = x.to(tl.float16)
        w_high = tl.load(High + cols[None, :] * K + k[:, None], (cols[None, :] < N) & (k[:, None] < K), other=0)
        primary = _rounding_boundary(tl.dot(x_high, w_high))
        if INPUT_PROTECTED:
            correction = tl.dot(x_low, w_high)
        else:
            w_low = tl.load(Low + cols[None, :] * K + k[:, None], (cols[None, :] < N) & (k[:, None] < K), other=0)
            correction = tl.dot(x_high, w_low)
        block = _rounding_boundary(primary + correction * (1 / 2048.0))
        accumulator += block * (1 / 256.0)
    if HAS_BIAS:
        accumulator += tl.load(Bias + cols, cols < N, other=0)[None, :]
    accumulator = _rounding_boundary(accumulator)
    if TEMPORAL:
        frame = rows % T
        spatial = rows // T % (H * W)
        batch = rows // (T * H * W)
    else:
        frame = tl.full((BM,), 0, tl.int32)
        spatial = rows % (H * W)
        batch = rows // (H * W)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    residual_offset = batch[:, None] * S0 + cols[None, :] * S1 + frame[:, None] * ST + spatial[:, None] // W * SH + spatial[:, None] % W * SW
    skip = tl.load(Residual + residual_offset, valid, other=0).to(tl.float32)
    result = accumulator + skip
    output_offset = ((batch[:, None] * N + cols[None, :]) * T + frame[:, None]) * (H * W) + spatial[:, None]
    tl.store(Output + output_offset, result, valid)


def fused(value,parts,bias,residual,*,config,output=None,inspect=False):
    assert not torch.is_grad_enabled() and not torch.is_autocast_enabled()
    assert value.is_cuda and value.dtype==torch.float32 and value.is_contiguous()
    n,k=parts[0].shape;m=value.numel()//k
    assert value.shape[-1]==k and residual.shape[1]==n and residual.numel()==m*n
    assert residual.dtype==torch.float32 and residual.device==value.device and residual.ndim in (4,5)
    assert len(parts)==2 and all(p.shape==(n,k) and p.dtype==torch.float16 and p.device==value.device and p.is_contiguous() for p in parts)
    assert bias is None or (bias.shape==(n,) and bias.dtype==torch.float32 and bias.device==value.device and bias.is_contiguous())
    if residual.ndim==4:
        t,h,w=1,*residual.shape[2:];s0,s1,sh,sw=residual.stride();st=0
    else:t,h,w=residual.shape[2:];s0,s1,st,sh,sw=residual.stride()
    if output is None:output=torch.empty(tuple(residual.shape),device=value.device,dtype=torch.float32)
    assert output.shape==residual.shape and output.is_contiguous() and output.dtype==torch.float32 and output.device==value.device
    bm,bn,group,warps,stages=config
    kernel=_out_projection_residual[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](value,*parts,parts[0] if bias is None else bias,output,residual,m,n,k,bias is not None,True,bm,bn,64,group,t,h,w,s0,s1,st,sh,sw,residual.ndim==5,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return (output,kernel) if inspect else output
