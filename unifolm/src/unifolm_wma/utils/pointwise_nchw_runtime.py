"""Stage/shape-guarded pointwise convolution; decision and heads stay native."""
import torch
from .triton_half_compensated_linear import pack_half_weight
from .triton_pointwise_nchw import pointwise_nchw


# (batch, input channels, output channels, H, W) -> (BM, groupM).
SELECTED = {
    (16, 320, 640, 20, 32):(64, 32),
    (16, 640, 1280, 10, 16):(64, 32),
    (16, 640, 320, 40, 64):(64, 32),
    (16, 960, 640, 20, 32):(64, 8),
    (16, 960, 320, 40, 64):(64, 32),
    (16, 1280, 640, 20, 32):(64, 8),
    (16, 1920, 1280, 10, 16):(64, 32),
    (16, 1920, 640, 20, 32):(64, 8),
    (16, 2560, 1280, 5, 8):(32, 32),
    (16, 2560, 1280, 10, 16):(64, 32),
}


def eligible(module):
    pairs={(shape[1],shape[2]) for shape in SELECTED}
    return (isinstance(module,torch.nn.Conv2d) and module.kernel_size==(1,1)
        and module.stride==(1,1) and module.padding==(0,0)
        and module.dilation==(1,1) and module.groups==1 and module.padding_mode=='zeros'
        and (module.in_channels,module.out_channels) in pairs)


def make_forward(module):
    original=module.forward
    signature,parts=None,None

    def forward(value):
        nonlocal signature,parts
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or not value.is_cuda or value.dtype!=torch.float32 or value.ndim!=4
                or not value.is_contiguous() or not eligible(module)
                or module.weight.dtype!=torch.float32 or not module.weight.is_contiguous()
                or module.weight.device!=value.device
                or (module.bias is not None and (module.bias.dtype!=torch.float32
                    or module.bias.device!=value.device or not module.bias.is_contiguous()))):
            return original(value)
        shape=(value.shape[0],value.shape[1],module.out_channels,*value.shape[2:])
        if shape not in SELECTED:return original(value)
        weight=module.weight
        try:version=weight._version
        except RuntimeError:return original(value)
        key=(weight.data_ptr(),version,tuple(weight.shape),weight.dtype,weight.device)
        if key!=signature:
            if torch.cuda.is_current_stream_capturing():return original(value)
            matrix=weight.view(module.out_channels,module.in_channels)
            reuse=(parts if parts is not None and parts[0].shape==matrix.shape
                   and parts[0].device==weight.device else None)
            parts=pack_half_weight(matrix,out=reuse)
            signature=key
        block_m,group=SELECTED[shape]
        return pointwise_nchw(value,parts,module.bias,block_m=block_m,group_m=group)

    return forward


def install_pointwise_nchw(diffusion):
    count=0
    for module in diffusion.modules():
        if eligible(module):
            if getattr(module,'_pointwise_nchw_installed',False):
                raise ValueError('Pointwise NCHW already installed')
            module.forward=make_forward(module)
            module._pointwise_nchw_installed=True
            count+=1
    return count
