import torch
import triton
import triton.language as tl
from unifolm_wma.utils.triton_compensated_linear import _rounding_boundary
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.gemm_output_allocation import empty as empty_gemm_output

@triton.jit
def _presplit_two_product_gemm(X, XL, High, Low, Bias, Output, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr, INPUT_PROTECTED: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
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
        input_offset = rows[:, None] * K + k[None, :]
        input_valid = (rows[:, None] < M) & (k[None, :] < K)
        x_high = tl.load(X + input_offset, input_valid, other=0)
        x_low = tl.load(XL + input_offset, input_valid, other=0)
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
    tl.store(Output + rows[:, None] * N + cols[None, :], accumulator, (rows[:, None] < M) & (cols[None, :] < N))


def fused(value,parts,bias=None,*,config,output=None,input_parts=None,inspect=False):
    assert not torch.is_grad_enabled() and not torch.is_autocast_enabled()
    assert value.is_cuda and value.dtype==torch.float32 and value.is_contiguous() and value.ndim>=1
    n,k=parts[0].shape;m=value.numel()//k
    assert value.shape[-1]==k and len(parts)==2 and all(p.shape==(n,k) and p.dtype==torch.float16 and p.device==value.device and p.is_contiguous() for p in parts)
    assert bias is None or (bias.shape==(n,) and bias.dtype==torch.float32 and bias.device==value.device and bias.is_contiguous())
    packed=pack_half_weight(value.view(m,k),weight_scale=1.,out=input_parts)
    if output is None:output=empty_gemm_output('classic',(*value.shape[:-1],n),device=value.device,dtype=torch.float32)
    assert output.numel()==m*n and output.dtype==torch.float32 and output.device==value.device and output.is_contiguous()
    bm,bn,group,warps,stages=config
    kernel=_presplit_two_product_gemm[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](*packed,*parts,parts[0] if bias is None else bias,output,m,n,k,bias is not None,True,bm,bn,64,group,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return (output,packed,kernel) if inspect else output
