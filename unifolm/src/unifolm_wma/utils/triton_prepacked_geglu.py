"""Diagnostic fusion of the accepted BK64 compensated projection and GEGLU."""
import torch; from .triton_zero_weight_low_geglu import run as run_zero_geglu, residual_count as refresh_geglu_count
from .geglu_output_allocation import empty as empty_geglu_output
import triton
import triton.language as tl
from .triton_compensated_linear import _rounding_boundary
from .triton_half_compensated_linear import pack_half_weight


@triton.jit
def _short_product(xh,xl,wh,wl):
    primary = _rounding_boundary(tl.dot(xh,wh))
    correction = tl.dot(xl,wh)
    correction = tl.dot(xh,wl,correction)
    return _rounding_boundary(primary+correction*(1./2048.))*(1./256.)


@triton.jit
def _prepacked_geglu(XH,XL,WH,WL,Bias,Y,
                      M:tl.constexpr,C:tl.constexpr,K:tl.constexpr,
                      HAS_BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,
                      GROUP_M:tl.constexpr):
    pid = tl.program_id(0)
    row_tiles,col_tiles = tl.cdiv(M,BM),tl.cdiv(C,BN)
    group = pid//(GROUP_M*col_tiles)
    first_row = group*GROUP_M
    group_rows = tl.minimum(GROUP_M,row_tiles-first_row)
    local = pid%(GROUP_M*col_tiles)
    rows = (first_row+local%group_rows)*BM+tl.arange(0,BM)
    cols = (local//group_rows)*BN+tl.arange(0,BN)
    inner = tl.arange(0,64)
    value = tl.full((BM,BN),0.,tl.float32)
    gate = tl.full((BM,BN),0.,tl.float32)
    for start in range(tl.cdiv(K,64)):
        k = start*64+inner
        input_offset = rows[:,None]*K+k[None,:]
        input_valid = (rows[:,None]<M)&(k[None,:]<K)
        xh = tl.load(XH+input_offset,input_valid,other=0)
        xl = tl.load(XL+input_offset,input_valid,other=0)
        offset = cols[None,:]*K+k[:,None]
        valid = (cols[None,:]<C)&(k[:,None]<K)
        vh = tl.load(WH+offset,valid,other=0)
        vl = tl.load(WL+offset,valid,other=0)
        gh = tl.load(WH+offset+C*K,valid,other=0)
        gl = tl.load(WL+offset+C*K,valid,other=0)
        value += _short_product(xh,xl,vh,vl)
        gate += _short_product(xh,xl,gh,gl)
    if HAS_BIAS:
        value += tl.load(Bias+cols,cols<C,other=0)[None,:]
        gate += tl.load(Bias+cols+C,cols<C,other=0)[None,:]
    gelu = (gate*.5)*(1.+tl.math.erf(gate*.7071067811865476))
    tl.store(Y+rows[:,None]*C+cols[None,:],value*gelu,
             (rows[:,None]<M)&(cols[None,:]<C))


def prepacked_geglu(value,parts,bias=None,*,block_m=64,block_n=64,
                     group_m=32,num_warps=4,num_stages=2):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim<1 or not value.is_contiguous()):
        raise ValueError('Expected contiguous inference CUDA FP32 input')
    if len(parts)!=2 or any(p.ndim!=2 or p.shape!=parts[0].shape
            or p.shape[1]!=value.shape[-1] or p.shape[0]%2
            or p.dtype!=torch.float16 or p.device!=value.device
            or not p.is_contiguous() for p in parts):
        raise ValueError('Invalid GEGLU compensated weights')
    channels,inner = parts[0].shape[0]//2,value.shape[-1]
    rows = value.numel()//inner
    if min(rows,channels,inner)<=0:
        raise ValueError('Empty shapes unsupported')
    if bias is not None and (bias.shape!=(2*channels,) or bias.dtype!=torch.float32
            or bias.device!=value.device or not bias.is_contiguous()):
        raise ValueError('Invalid GEGLU FP32 bias')
    if (block_m not in {32,64} or block_n not in {32,64}
            or group_m not in {8,16,32} or num_warps not in {4,8}
            or num_stages not in {2,3}):
        raise ValueError('Unsupported GEGLU tile')
    packed = pack_half_weight(value.view(rows,inner),weight_scale=1.)
    output = empty_geglu_output((*value.shape[:-1],channels),device=value.device,dtype=torch.float32)
    if not run_zero_geglu(packed,parts,bias,output,output,output,rows,channels,inner,block_m,block_n,group_m,False,False,False,64,num_warps,num_stages):
        _prepacked_geglu[(triton.cdiv(rows,block_m)*triton.cdiv(channels,block_n),)](
            *packed,*parts,parts[0] if bias is None else bias,output,
            rows,channels,inner,bias is not None,block_m,block_n,group_m,
            num_warps=num_warps,num_stages=num_stages,enable_fp_fusion=False)
    return output


def make_forward(module):
    original = module.forward
    projection = module.proj
    signature, parts = None, None
    selected = {(10240,640),(2560,1280),(640,1280)}

    def forward(value):
        nonlocal signature, parts
        if (module.training or projection.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or not value.is_cuda
                or value.dtype!=torch.float32 or value.ndim<1
                or not value.is_contiguous() or value.shape[-1]!=projection.in_features
                or (value.numel()//projection.in_features,projection.in_features) not in selected
                or projection.out_features!=8*projection.in_features
                or projection.weight.dtype!=torch.float32
                or projection.weight.device!=value.device
                or not projection.weight.is_contiguous()
                or (projection.bias is not None and (projection.bias.dtype!=torch.float32
                    or projection.bias.device!=value.device or not projection.bias.is_contiguous()))):
            return original(value)
        weight = projection.weight
        try:
            version = weight._version
        except RuntimeError:
            return original(value)
        key = weight.data_ptr(),version,tuple(weight.shape),weight.dtype,weight.device
        if signature!=key:
            if torch.cuda.is_current_stream_capturing():
                return original(value)
            reuse = (parts if parts is not None and parts[0].shape==weight.shape
                     and parts[0].device==weight.device else None)
            parts = pack_half_weight(weight,out=reuse)
            refresh_geglu_count(parts,refresh=True)
            signature = key
        return prepacked_geglu(value,parts,projection.bias,block_m=64,
            block_n=64 if projection.in_features==640 else 32,
            group_m=32,num_warps=4,num_stages=2)

    return forward


def install_prepacked_geglus(model):
    from unifolm_wma.modules.attention import GEGLU
    count = 0
    for module in model.modules():
        if (isinstance(module,GEGLU) and module.proj.in_features in {640,1280}
                and module.proj.out_features==8*module.proj.in_features):
            if (getattr(module,'_prepacked_geglu_installed',False)
                    or getattr(module,'_compensated_geglu_installed',False)
                    or not getattr(module.proj,'_tf32x3_installed',False)):
                raise ValueError('Wide GEGLU must replace exactly one accepted compensated projection')
            module.forward = make_forward(module)
            module._prepacked_geglu_installed = True
            count += 1
    return count
