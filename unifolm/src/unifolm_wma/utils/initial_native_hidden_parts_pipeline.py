import torch
import triton
import triton.language as tl
from .triton_initial_bk32_geglu import _piece32
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary
from .triton_native_geglu_activation import native_geglu
from .triton_prepacked_input_linear import _prepacked_gemm
from .geglu_output_allocation import empty as empty_geglu
from .gemm_output_allocation import empty as empty_gemm

@triton.jit
def _bk32_native_hidden_parts(X, WH, WL, Bias, YH, YL, V, G,
                       M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                       HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr, GROUP: tl.constexpr, DEBUG: tl.constexpr):
    pid = tl.program_id(0)
    mt, nt = tl.cdiv(M, BM), tl.cdiv(C, BN)
    group = pid // (GROUP * nt)
    first = group * GROUP
    size = tl.minimum(GROUP, mt - first)
    local = pid % (GROUP * nt)
    rows = (first + local % size) * BM + tl.arange(0, BM)
    cols = (local // size) * BN + tl.arange(0, BN)
    inner = tl.arange(0, 32)
    value = tl.full((BM, BN), 0., tl.float32)
    gate = tl.full((BM, BN), 0., tl.float32)
    for block in range(tl.cdiv(K, 32)):
        k = block * 32 + inner
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        xh, xl = _half_split(x, 1.)
        offset = cols[None, :] * K + k[:, None]
        valid = (cols[None, :] < C) & (k[:, None] < K)
        vh = tl.load(WH + offset, valid, other=0)
        vl = tl.load(WL + offset, valid, other=0)
        gh = tl.load(WH + offset + C * K, valid, other=0)
        gl = tl.load(WL + offset + C * K, valid, other=0)
        value += _piece32(xh, xl, vh, vl)
        gate += _piece32(xh, xl, gh, gl)
    if HAS_BIAS:
        value += tl.load(Bias + cols, cols < C, other=0)[None, :]
        gate += tl.load(Bias + cols + C, cols < C, other=0)[None, :]
    output = rows[:, None] * C + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < C)
    if DEBUG:
        tl.store(V + output, value, valid)
        tl.store(G + output, gate, valid)
    hidden = _rounding_boundary(native_geglu(value, gate))
    high, low = _half_split(hidden, 1.)
    tl.store(YH + output, high, valid)
    tl.store(YL + output, low, valid)


def initial_native_hidden_parts(value,parts,bias=None):
    assert value.is_cuda and value.dtype==torch.float32 and value.ndim==2 and value.is_contiguous()
    rows,inner=value.shape;channels=parts[0].shape[0]//2
    assert inner==512 and len(parts)==2
    assert all(p.shape==(channels*2,inner) and p.dtype==torch.float16 and p.device==value.device and p.is_contiguous() for p in parts)
    outputs=tuple(empty_geglu((rows,channels),device=value.device,dtype=torch.float16) for _ in range(2))
    _bk32_native_hidden_parts[(triton.cdiv(rows,64)*triton.cdiv(channels,64),)](
        value,*parts,parts[0] if bias is None else bias,*outputs,outputs[0],outputs[0],
        rows,channels,inner,bias is not None,64,64,32,False,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return outputs


def initial_packed_final(inputs,parts,bias=None):
    high,low=inputs;rows,inner=high.shape;channels=parts[0].shape[0]
    assert high.dtype==low.dtype==torch.float16 and high.shape==low.shape and inner==2048
    output=empty_gemm('classic',(rows,channels),device=high.device,dtype=torch.float32)
    _prepacked_gemm[(triton.cdiv(rows,64)*triton.cdiv(channels,128),)](
        high,low,*parts,parts[0] if bias is None else bias,output,
        rows,channels,inner,bias is not None,64,128,32,32,1,
        num_warps=4,num_stages=3,enable_fp_fusion=False)
    return output
