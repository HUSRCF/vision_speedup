"""Diagnostic three-product temporal convolution with FP32 master/output."""
import os
import torch
from .conv_buffer_allocation import empty as empty_conv_buffer
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split, pack_half_weight
from .triton_compensated_linear import _rounding_boundary


@triton.jit
def _pack_temporal(X, High, Low, M: tl.constexpr, C: tl.constexpr,
                   T: tl.constexpr, S: tl.constexpr, BR: tl.constexpr,
                   BC: tl.constexpr):
    rows = tl.program_id(0)*BR + tl.arange(0, BR)[:, None]
    k = tl.program_id(1)*BC + tl.arange(0, BC)[None, :]
    batch = rows//(T*S)
    frame, spatial = (rows//S)%T, rows%S
    channel, tap = k//3, k%3
    source_frame = frame + tap - 1
    valid = (rows < M) & (k < C*3)
    source = ((batch*C+channel)*T+source_frame)*S+spatial
    value = tl.load(X+source, valid & (source_frame >= 0) & (source_frame < T), other=0)
    high, low = _half_split(value, 1.)
    output = rows*(C*3)+k
    tl.store(High+output, high, valid)
    tl.store(Low+output, low, valid)


@triton.jit
def _temporal_gemm(XH, XL, WH, WL, Bias, Output,
                    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    POSITIONS: tl.constexpr, HAS_BIAS: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    GROUP_M: tl.constexpr, FUSED_OUTPUT: tl.constexpr):
    pid = tl.program_id(0)
    row_tiles, col_tiles = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid//(GROUP_M*col_tiles)
    first_row = group*GROUP_M
    group_rows = tl.minimum(GROUP_M, row_tiles-first_row)
    local = pid%(GROUP_M*col_tiles)
    rows = (first_row+local%group_rows)*BM+tl.arange(0, BM)
    cols = (local//group_rows)*BN+tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0., tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start*BK+inner
        input_offset = rows[:, None]*K+k[None, :]
        input_mask = (rows[:, None] < M) & (k[None, :] < K)
        xh = tl.load(XH+input_offset, input_mask, other=0)
        xl = tl.load(XL+input_offset, input_mask, other=0)
        weight_offset = cols[None, :]*K+k[:, None]
        weight_mask = (cols[None, :] < N) & (k[:, None] < K)
        wh = tl.load(WH+weight_offset, weight_mask, other=0)
        wl = tl.load(WL+weight_offset, weight_mask, other=0)
        primary = _rounding_boundary(tl.dot(xh, wh))
        correction = tl.dot(xl, wh)
        correction = tl.dot(xh, wl, correction)
        block = _rounding_boundary(primary+correction*(1./2048.))
        acc += block*(1./256.)
    if HAS_BIAS:
        acc += tl.load(Bias+cols, cols < N, other=0)[None, :]
    if FUSED_OUTPUT:
        output_offset = ((rows[:, None]//POSITIONS)*N+cols[None, :])*POSITIONS+rows[:, None]%POSITIONS
    else:
        output_offset = rows[:, None]*N+cols[None, :]
    tl.store(Output+output_offset, acc, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _restore_layout(Matrix, Output, M: tl.constexpr, N: tl.constexpr,
                     POSITIONS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < M*N
    batch, channel = i//(N*POSITIONS), (i//POSITIONS)%N
    row = batch*POSITIONS+i%POSITIONS
    value = tl.load(Matrix+row*N+channel, valid, other=0)
    tl.store(Output+i, value, valid)


def temporal_weight_parts(weight, *, out=None):
    if weight.ndim != 5 or weight.shape[2:] != (3, 1, 1):
        raise ValueError('Expected OI311 temporal weight')
    return pack_half_weight(weight.view(weight.shape[0], -1), out=out)


def compensated_temporal_conv(value, parts, bias=None, *, block_m=32,
                               block_n=64, block_k=64, fused_output=True):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype != torch.float32
            or value.ndim != 5 or not value.is_contiguous()):
        raise ValueError('Expected contiguous inference CUDA FP32 NCTHW')
    batch, inputs, frames, height, width = value.shape
    if len(parts) != 2 or any(p.ndim != 2 or p.shape != parts[0].shape
            or p.shape[1] != inputs*3 or p.dtype != torch.float16
            or p.device != value.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid compensated temporal weights')
    outputs = parts[0].shape[0]
    if min(batch, inputs, outputs, frames, height, width) <= 0:
        raise ValueError('Empty dimensions are unsupported')
    if bias is not None and (bias.shape != (outputs,) or bias.dtype != torch.float32
            or bias.device != value.device or not bias.is_contiguous()):
        raise ValueError('Invalid temporal FP32 bias')
    if block_m not in {32,64} or block_n not in {64,128} or block_k not in {32,64,128}:
        raise ValueError('Unsupported temporal GEMM tile')
    spatial, positions = height*width, frames*height*width
    rows, reduction = batch*positions, inputs*3
    packed = tuple(empty_conv_buffer('temporal',(rows, reduction), device=value.device,
                               dtype=torch.float16) for _ in range(2))
    _pack_temporal[(triton.cdiv(rows,32), triton.cdiv(reduction,32))](
        value, *packed, rows, inputs, frames, spatial, 32,32,
        num_warps=4, enable_fp_fusion=False)
    result = empty_conv_buffer('temporal',(batch,outputs,frames,height,width), device=value.device,
                         dtype=torch.float32)
    matrix = result if fused_output else empty_conv_buffer('temporal',(rows,outputs),
                         device=value.device, dtype=torch.float32)
    _temporal_gemm[(triton.cdiv(rows,block_m)*triton.cdiv(outputs,block_n),)](
        *packed, *parts, parts[0] if bias is None else bias, matrix,
        rows, outputs, reduction, positions, bias is not None,
        block_m,block_n,block_k,32,fused_output,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    if not fused_output:
        _restore_layout[(triton.cdiv(rows*outputs,256),)](
            matrix,result,rows,outputs,positions,256,num_warps=4)
    return result


def eligible_temporal(module):
    return (isinstance(module, torch.nn.Conv3d)
            and module.in_channels == module.out_channels
            and module.in_channels in {320,640,1280}
            and module.kernel_size == (3,1,1) and module.stride == (1,1,1)
            and module.padding == (1,0,0) and module.dilation == (1,1,1)
            and module.groups == 1 and module.padding_mode == 'zeros')


def make_compensated_temporal_forward(module):
    implicit_mode = os.environ.get('UNIFOLM_IMPLICIT_TEMPORAL_CONV', '0')
    if implicit_mode not in {'0', '1'}:
        raise ValueError('UNIFOLM_IMPLICIT_TEMPORAL_CONV must be 0 or 1')
    original = module.forward
    parts = None
    signature = None
    selected = {(320,40,64),(640,20,32),(1280,10,16),(1280,5,8)}

    def forward(value):
        nonlocal parts, signature
        if (not eligible_temporal(module) or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or not value.is_cuda
                or value.dtype != torch.float32 or value.ndim != 5
                or value.shape[0] != 1 or value.shape[2] != 16
                or (value.shape[1],*value.shape[-2:]) not in selected
                or not value.is_contiguous() or torch.backends.cudnn.allow_tf32
                or module.weight.dtype != torch.float32
                or module.weight.device != value.device or not module.weight.is_contiguous()
                or (module.bias is not None and (module.bias.dtype != torch.float32
                    or module.bias.device != value.device or not module.bias.is_contiguous()))):
            return original(value)
        try:
            version = module.weight._version
        except RuntimeError:
            return original(value)
        weight = module.weight
        key = weight.data_ptr(),version,tuple(weight.shape),weight.dtype,weight.device
        if signature != key:
            if torch.cuda.is_current_stream_capturing():
                return original(value)
            reuse = parts if parts is not None and parts[0].device == weight.device else None
            parts = temporal_weight_parts(weight,out=reuse)
            signature = key
        if implicit_mode == '1' and value.shape[1] in {320,640}:
            from .triton_implicit_temporal_conv import implicit_temporal_conv
            return implicit_temporal_conv(value,parts,module.bias,
                block_m=64,block_n=64,
                group_m=8 if value.shape[1] == 320 else 16,num_stages=2)
        return compensated_temporal_conv(value,parts,module.bias,
            block_m=64,block_n=64,block_k=64,fused_output=True)

    return forward
