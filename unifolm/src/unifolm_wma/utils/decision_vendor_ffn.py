"""Opt-in vendor compensation for decision FFN output projections only."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .cublas_compensated_linear import cublas_compensated_linear
from .cublas_chunked_compensated_linear import chunked_compensated_linear


def make_forward(module,mode):
    if mode not in {'full','chunked'}:raise ValueError('Unknown vendor FFN mode')
    original = module.forward
    signature,parts = None,None
    selected = {(10240,2560,640),(2560,5120,1280),(640,5120,1280)}

    def forward(value):
        nonlocal signature,parts
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or not torch.backends.cudnn.allow_tf32
                or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype!=torch.float32 or value.ndim<1
                or not value.is_contiguous() or value.shape[-1]!=module.in_features
                or (value.numel()//module.in_features,module.in_features,module.out_features) not in selected
                or module.weight.dtype!=torch.float32 or module.weight.device!=value.device
                or not module.weight.is_contiguous()
                or (module.bias is not None and (module.bias.dtype!=torch.float32
                    or module.bias.device!=value.device or not module.bias.is_contiguous()))):
            return original(value)
        weight = module.weight
        try:version = weight._version
        except RuntimeError:return original(value)
        key = weight.data_ptr(),version,tuple(weight.shape),weight.dtype,weight.device
        if key!=signature:
            if torch.cuda.is_current_stream_capturing():return original(value)
            reuse = (parts if parts is not None and parts[0].shape==weight.shape
                     and parts[0].device==weight.device else None)
            parts = pack_half_weight(weight,out=reuse)
            signature = key
        if mode=='full':
            result = cublas_compensated_linear(value,parts,module.bias)
        else:
            result = chunked_compensated_linear(value,parts,module.bias,
                chunk_k=512 if module.in_features==2560 else 1024)
        return original(value) if result is None else result

    return forward


def install_decision_vendor_ffns(diffusion,mode):
    count = 0
    for name,module in diffusion.named_modules():
        if (not name.endswith('.ff.net.2') or not isinstance(module,torch.nn.Linear)
                or module.out_features not in {640,1280}
                or module.in_features!=4*module.out_features):
            continue
        if (getattr(module,'_decision_vendor_ffn_installed',False)
                or not getattr(module,'_tf32x3_installed',False)):
            raise ValueError('Vendor FFN requires an accepted compensated output Linear')
        module.forward = make_forward(module,mode)
        module._decision_vendor_ffn_installed = True
        count += 1
    return count
