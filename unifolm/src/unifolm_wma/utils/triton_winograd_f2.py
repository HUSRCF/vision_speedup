"""Diagnostic F(2,3): binary-exact transform coefficients and FP32 outputs.

All transforms use FP32. Component GEMMs retain the accepted three FP16
products with short FP32 accumulation. Installation requires an explicit opt-in.
"""
import torch; from .triton_zero_weight_low_spatial import run_packed_components, run_f2_components, residual_counts
from .conv_buffer_allocation import empty as empty_conv_buffer
import triton
import triton.language as tl
from .triton_half_compensated_linear import _half_split
from .triton_prepacked_input_linear import _prepacked_gemm


@triton.jit
def _g2(a, b, c):
    return a, (a+b+c)*.5, (a-b+c)*.5, c


@triton.jit
def _bt2(a, b, c, d):
    return a-c, b+c, c-b, b-d


@triton.jit
def _at2(a, b, c, d):
    return a+b+c, b-c-d


@triton.jit
def _store4_parts(High, Low, offset, stride: tl.constexpr,
                  a, b, c, d, valid, SCALE: tl.constexpr):
    ah, al = _half_split(a, SCALE)
    bh, bl = _half_split(b, SCALE)
    ch, cl = _half_split(c, SCALE)
    dh, dl = _half_split(d, SCALE)
    tl.store(High+offset, ah, valid)
    tl.store(Low+offset, al, valid)
    tl.store(High+offset+stride, bh, valid)
    tl.store(Low+offset+stride, bl, valid)
    tl.store(High+offset+2*stride, ch, valid)
    tl.store(Low+offset+2*stride, cl, valid)
    tl.store(High+offset+3*stride, dh, valid)
    tl.store(Low+offset+3*stride, dl, valid)


@triton.jit
def _filter_f2(W, High, Low, PAIRS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < PAIRS
    a0 = tl.load(W+i*9, valid, other=0)
    a1 = tl.load(W+i*9+1, valid, other=0)
    a2 = tl.load(W+i*9+2, valid, other=0)
    b0 = tl.load(W+i*9+3, valid, other=0)
    b1 = tl.load(W+i*9+4, valid, other=0)
    b2 = tl.load(W+i*9+5, valid, other=0)
    c0 = tl.load(W+i*9+6, valid, other=0)
    c1 = tl.load(W+i*9+7, valid, other=0)
    c2 = tl.load(W+i*9+8, valid, other=0)
    a0, a1, a2, a3 = _g2(a0, a1, a2)
    b0, b1, b2, b3 = _g2(b0, b1, b2)
    c0, c1, c2, c3 = _g2(c0, c1, c2)
    v0, v1, v2, v3 = _g2(a0, b0, c0)
    _store4_parts(High, Low, i, 4*PAIRS, v0, v1, v2, v3, valid, 256.)
    v0, v1, v2, v3 = _g2(a1, b1, c1)
    _store4_parts(High, Low, i+PAIRS, 4*PAIRS, v0, v1, v2, v3, valid, 256.)
    v0, v1, v2, v3 = _g2(a2, b2, c2)
    _store4_parts(High, Low, i+2*PAIRS, 4*PAIRS, v0, v1, v2, v3, valid, 256.)
    v0, v1, v2, v3 = _g2(a3, b3, c3)
    _store4_parts(High, Low, i+3*PAIRS, 4*PAIRS, v0, v1, v2, v3, valid, 256.)


@triton.jit
def _load_row(X, origin, valid, x, W: tl.constexpr):
    a = tl.load(X+origin, valid & (x>=0) & (x<W), other=0)
    b = tl.load(X+origin+1, valid & (x+1>=0) & (x+1<W), other=0)
    c = tl.load(X+origin+2, valid & (x+2>=0) & (x+2<W), other=0)
    d = tl.load(X+origin+3, valid & (x+3>=0) & (x+3<W), other=0)
    return _bt2(a, b, c, d)


@triton.jit
def _input_f2(X, High, Low, PAIRS: tl.constexpr, C: tl.constexpr,
              H: tl.constexpr, W: tl.constexpr, TH: tl.constexpr,
              TW: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    tile, channel = i//C, i%C
    batch, local = tile//(TH*TW), tile%(TH*TW)
    y, x = (local//TW)*2-1, (local%TW)*2-1
    origin = ((batch*C+channel)*H+y)*W+x
    valid = i < PAIRS
    a0, a1, a2, a3 = _load_row(X, origin, valid & (y>=0) & (y<H), x, W)
    b0, b1, b2, b3 = _load_row(X, origin+W, valid & (y+1>=0) & (y+1<H), x, W)
    c0, c1, c2, c3 = _load_row(X, origin+2*W, valid & (y+2>=0) & (y+2<H), x, W)
    d0, d1, d2, d3 = _load_row(X, origin+3*W, valid & (y+3>=0) & (y+3<H), x, W)
    v0, v1, v2, v3 = _bt2(a0, b0, c0, d0)
    _store4_parts(High, Low, i, 4*PAIRS, v0, v1, v2, v3, valid, 1.)
    v0, v1, v2, v3 = _bt2(a1, b1, c1, d1)
    _store4_parts(High, Low, i+PAIRS, 4*PAIRS, v0, v1, v2, v3, valid, 1.)
    v0, v1, v2, v3 = _bt2(a2, b2, c2, d2)
    _store4_parts(High, Low, i+2*PAIRS, 4*PAIRS, v0, v1, v2, v3, valid, 1.)
    v0, v1, v2, v3 = _bt2(a3, b3, c3, d3)
    _store4_parts(High, Low, i+3*PAIRS, 4*PAIRS, v0, v1, v2, v3, valid, 1.)


@triton.jit
def _components_f2(VH, VL, UH, UL, Product, ROWS: tl.constexpr,
                   N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, BK: tl.constexpr):
    part = tl.program_id(2)
    _prepacked_gemm(VH+part*ROWS*K, VL+part*ROWS*K,
        UH+part*N*K, UL+part*N*K, UH, Product+part*ROWS*N,
        ROWS, N, K, False, BM, BN, BK, 0)


@triton.jit
def _output_column(Product, offset, PAIRS: tl.constexpr, valid):
    a = tl.load(Product+offset, valid, other=0)
    b = tl.load(Product+offset+4*PAIRS, valid, other=0)
    c = tl.load(Product+offset+8*PAIRS, valid, other=0)
    d = tl.load(Product+offset+12*PAIRS, valid, other=0)
    return _at2(a, b, c, d)


@triton.jit
def _output_f2(Product, Bias, Y, PAIRS: tl.constexpr, C: tl.constexpr,
               H: tl.constexpr, W: tl.constexpr, TH: tl.constexpr,
               TW: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < PAIRS
    a0, a1 = _output_column(Product, i, PAIRS, valid)
    b0, b1 = _output_column(Product, i+PAIRS, PAIRS, valid)
    c0, c1 = _output_column(Product, i+2*PAIRS, PAIRS, valid)
    d0, d1 = _output_column(Product, i+3*PAIRS, PAIRS, valid)
    top0, top1 = _at2(a0, b0, c0, d0)
    bottom0, bottom1 = _at2(a1, b1, c1, d1)
    tile, channel = i//C, i%C
    batch, local = tile//(TH*TW), tile%(TH*TW)
    y, x = (local//TW)*2, (local%TW)*2
    origin = ((batch*C+channel)*H+y)*W+x
    bias = tl.full((BLOCK,), 0, tl.float32)
    if HAS_BIAS:
        bias = tl.load(Bias+channel, valid, other=0)
    tl.store(Y+origin, top0+bias, valid & (y<H) & (x<W))
    tl.store(Y+origin+1, top1+bias, valid & (y<H) & (x+1<W))
    tl.store(Y+origin+W, bottom0+bias, valid & (y+1<H) & (x<W))
    tl.store(Y+origin+W+1, bottom1+bias, valid & (y+1<H) & (x+1<W))


def transform_filter_f2(weight, *, out=None):
    if (not weight.is_cuda or weight.dtype != torch.float32 or weight.ndim != 4
            or weight.shape[-2:] != (3, 3) or not weight.is_contiguous()
            or torch.is_grad_enabled()):
        raise ValueError('F2 filter requires contiguous CUDA FP32 OI33 inference')
    outputs, inputs = weight.shape[:2]
    shape = (16, outputs, inputs)
    parts = tuple(torch.empty(shape, device=weight.device, dtype=torch.float16)
                  for _ in range(2)) if out is None else out
    if len(parts) != 2 or any(p.shape != shape or p.dtype != torch.float16
            or p.device != weight.device or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid F2 filter storage')
    _filter_f2[(triton.cdiv(outputs*inputs, 128),)](
        weight, *parts, outputs*inputs, 128, num_warps=4, enable_fp_fusion=False)
    residual_counts(parts,'f2',refresh=True)
    return parts


def winograd_f2_conv2d(value, weight_parts, bias=None, *, block_m=32,
                        block_n=64, block_k=64, num_warps=4, num_stages=2):
    if (not value.is_cuda or value.dtype != torch.float32 or value.ndim != 4
            or not value.is_contiguous() or torch.is_grad_enabled()):
        raise ValueError('F2 input requires contiguous CUDA FP32 NCHW inference')
    if len(weight_parts) != 2 or any(p.dtype != torch.float16 or p.ndim != 3
            or p.shape[0] != 16 or p.shape[2] != value.shape[1]
            or p.shape != weight_parts[0].shape or not p.is_contiguous()
            or p.device != value.device for p in weight_parts):
        raise ValueError('Invalid F2 transformed filters')
    batch, inputs, height, width = value.shape
    outputs = weight_parts[0].shape[1]
    if min(batch, inputs, outputs, height, width) <= 0:
        raise ValueError('Empty F2 dimensions')
    if bias is not None and (bias.shape != (outputs,) or bias.device != value.device
            or bias.dtype != torch.float32 or not bias.is_contiguous()):
        raise ValueError('Invalid F2 bias')
    if (block_m not in {32, 64} or block_n != 64 or block_k not in {32, 64}
            or num_warps != 4 or num_stages not in {2, 3}):
        raise ValueError('Unsupported F2 kernel configuration')
    th, tw = triton.cdiv(height, 2), triton.cdiv(width, 2)
    tiles = batch*th*tw
    input_parts = tuple(empty_conv_buffer('f2',(16, tiles, inputs), device=value.device,
                                   dtype=torch.float16) for _ in range(2))
    _input_f2[(triton.cdiv(tiles*inputs, 128),)](value, *input_parts,
        tiles*inputs, inputs, height, width, th, tw, 128,
        num_warps=4, enable_fp_fusion=False)
    product = empty_conv_buffer('f2',(16, tiles, outputs), device=value.device, dtype=torch.float32)
    if not run_f2_components(input_parts,weight_parts,product,tiles,outputs,inputs,block_m,block_n,block_k,num_warps,num_stages):
        _components_f2[(triton.cdiv(tiles, block_m), triton.cdiv(outputs, block_n), 16)](
            *input_parts, *weight_parts, product, tiles, outputs, inputs, block_m,
            block_n, block_k, num_warps=num_warps, num_stages=num_stages,
            enable_fp_fusion=False)
    result = empty_conv_buffer('f2',(batch, outputs, height, width), device=value.device,
                         dtype=torch.float32)
    _output_f2[(triton.cdiv(tiles*outputs, 128),)](product,
        weight_parts[0] if bias is None else bias, result, tiles*outputs,
        outputs, height, width, th, tw, bias is not None, 128,
        num_warps=4, enable_fp_fusion=False)
    return result


def eligible_f2(module):
    return (isinstance(module, torch.nn.Conv2d) and module.kernel_size == (3, 3)
            and module.stride == (1, 1) and module.padding == (1, 1)
            and module.dilation == (1, 1) and module.groups == 1
            and module.padding_mode == 'zeros' and module.out_channels == 1280
            and module.in_channels in {1280, 2560})


def f2_spatial_selected(inputs, spatial, scope):
    if scope not in {'high', 'medium', 'deep'}:
        raise ValueError('F2 scope must be high, medium, or deep')
    if spatial == (10, 16):
        return inputs in {1280, 2560} and scope in {'high', 'medium'}
    return inputs == 1280 and spatial == (5, 8) and scope in {'high', 'deep'}


def make_f2_forward(module, *, scope='high'):
    f2_spatial_selected(module.in_channels, (10, 16), scope)
    original = module.forward
    parts = None
    signature = None

    def forward(value):
        nonlocal parts, signature
        if (not eligible_f2(module) or module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or not value.is_cuda
                or value.dtype != torch.float32 or value.ndim != 4
                or value.shape[:2] != (16, module.in_channels)
                or not f2_spatial_selected(module.in_channels, tuple(value.shape[2:]), scope)
                or not value.is_contiguous() or torch.backends.cudnn.allow_tf32
                or module.weight.dtype != torch.float32
                or not module.weight.is_contiguous() or module.weight.device != value.device
                or (module.bias is not None and (module.bias.dtype != torch.float32
                    or module.bias.device != value.device or not module.bias.is_contiguous()))):
            return original(value)
        try:
            version = module.weight._version
        except RuntimeError:
            return original(value)
        weight = module.weight
        key = weight.data_ptr(), version, tuple(weight.shape), weight.dtype, weight.device
        if signature != key:
            if torch.cuda.is_current_stream_capturing():
                return original(value)
            reuse = parts if parts is not None and parts[0].device == weight.device else None
            parts = transform_filter_f2(weight, out=reuse)
            signature = key
        return winograd_f2_conv2d(value, parts, module.bias, block_m=64)

    return forward


def install_f2_convolutions(model, *, scope='high'):
    count = 0
    for module in model.modules():
        if eligible_f2(module):
            if getattr(module, '_winograd_installed', False):
                raise ValueError('F2 cannot combine with another Winograd replacement')
            module.forward = make_f2_forward(module, scope=scope)
            module._winograd_f2_installed = True
            count += 1
    return count
