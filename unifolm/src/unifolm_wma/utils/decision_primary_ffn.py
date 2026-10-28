"""Decision FFN-only single Half product, with FP32 master/bias/output.

This deliberately changes arithmetic. World, attention, GEGLU, normalization,
feedback state and all other linears retain their accepted implementations.
Full-video quality must pass independently before this route can be adopted.
"""
import ctypes
import torch
import triton
import triton.language as tl
from .cublas_compensated_linear import _library,_READY,_stream_key,_gemm
from .triton_half_compensated_linear import pack_half_weight


@triton.jit
def _finish(Primary,Bias,SIZE:tl.constexpr,N:tl.constexpr,
             HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<SIZE
    result=tl.load(Primary+i,valid,other=0.)*(1./256.)
    if HAS_BIAS:result+=tl.load(Bias+i%N,valid,other=0.)
    tl.store(Primary+i,result,valid)


def primary_linear(value,high_weight,bias=None,*,debug_poison=False):
    if (torch.is_grad_enabled() or not value.is_cuda or value.dtype!=torch.float32
            or value.ndim<1 or not value.is_contiguous() or high_weight.ndim!=2
            or high_weight.dtype!=torch.float16 or high_weight.device!=value.device
            or not high_weight.is_contiguous() or value.shape[-1]!=high_weight.shape[1]):
        raise ValueError('Expected contiguous FP32 input and compensated high Half weights')
    rows=value.numel()//value.shape[-1]
    columns,inner=high_weight.shape
    if min(rows,columns,inner)<=0:raise ValueError('Empty GEMM unsupported')
    if bias is not None and (bias.shape!=(columns,) or bias.dtype!=torch.float32
            or bias.device!=value.device or not bias.is_contiguous()):
        raise ValueError('Invalid FP32 bias')
    with torch.cuda.device(value.device):
        key=_stream_key(value.device)
        if torch.cuda.is_current_stream_capturing() and key not in _READY:return None
        lib=_library()
        handle=ctypes.c_void_p(torch.cuda.current_blas_handle())
        mode=ctypes.c_int()
        status=lib.cublasGetPointerMode_v2(handle,ctypes.byref(mode))
        if status or mode.value!=0:raise RuntimeError('cuBLAS host pointer mode required')
        _READY.add(key)
        # Dynamic inputs are converted on every invocation/replay, inside timing.
        xh=value.view(rows,inner).to(torch.float16)
        primary=torch.empty((rows,columns),device=value.device,dtype=torch.float32)
        if debug_poison:primary.fill_(float('nan'))
        _gemm(lib,handle,xh,high_weight,primary)
        _finish[(triton.cdiv(rows*columns,256),)](primary,
            high_weight if bias is None else bias,rows*columns,columns,bias is not None,
            256,num_warps=4,enable_fp_fusion=False)
        return primary.view(*value.shape[:-1],columns)


def make_forward(module):
    original=module.forward
    signature,parts=None,None
    selected={(10240,2560,640),(2560,5120,1280),(640,5120,1280)}
    def forward(value):
        nonlocal signature,parts
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or not torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype!=torch.float32 or value.ndim<1
                or not value.is_contiguous() or value.shape[-1]!=module.in_features
                or (value.numel()//module.in_features,module.in_features,module.out_features) not in selected
                or module.weight.dtype!=torch.float32 or module.weight.device!=value.device
                or not module.weight.is_contiguous()
                or (module.bias is not None and (module.bias.dtype!=torch.float32
                    or module.bias.device!=value.device or not module.bias.is_contiguous()))):
            return original(value)
        weight=module.weight
        try:version=weight._version
        except RuntimeError:return original(value)
        key=weight.data_ptr(),version,tuple(weight.shape),weight.dtype,weight.device
        if key!=signature:
            if torch.cuda.is_current_stream_capturing():return original(value)
            reuse=parts if parts is not None and parts[0].shape==weight.shape and parts[0].device==weight.device else None
            parts=pack_half_weight(weight,out=reuse)
            signature=key
        result=primary_linear(value,parts[0],module.bias)
        return original(value) if result is None else result
    return forward


def install_decision_primary_ffns(diffusion,scope):
    if scope not in {'all','640','1280'}:raise ValueError('Unknown primary FFN scope')
    count=0
    for name,module in diffusion.named_modules():
        if (not name.endswith('.ff.net.2') or not isinstance(module,torch.nn.Linear)
                or module.out_features not in {640,1280}
                or module.in_features!=4*module.out_features
                or (scope!='all' and module.out_features!=int(scope))):continue
        if (not getattr(module,'_decision_vendor_ffn_installed',False)
                or getattr(module,'_decision_primary_ffn_installed',False)):
            raise ValueError('Primary FFN must wrap accepted decision vendor FFN')
        module.forward=make_forward(module)
        module._decision_primary_ffn_installed=True
        count+=1
    return count
