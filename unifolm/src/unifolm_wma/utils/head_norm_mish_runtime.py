"""Qualified head-only dispatch, retaining every original module and parameter."""
from collections import Counter
from contextlib import contextmanager

import torch
from unifolm_wma.utils.nvrtc_head_norm_mish import fused


class Controller:
    def __init__(self,model,enabled=False):
        self.enabled=enabled;self.originals=[];self.hits=Counter();self.fallbacks=Counter()
        for label in ('action','state'):
            head=getattr(model.model.diffusion_model,label+'_unet')
            blocks=[m for m in head.modules() if m.__class__.__name__=='Conv1dBlock']
            assert len(blocks)==37
            for block in blocks:
                conv,norm,activation=block.block
                assert type(conv) is torch.nn.Conv1d and type(norm) is torch.nn.GroupNorm and type(activation) is torch.nn.Mish
                assert not activation.inplace and norm.num_groups==8 and norm.affine
                original=block.forward;self.originals.append((block,original))
                def forward(value,*,block=block,conv=conv,norm=norm,activation=activation,original=original,label=label):
                    if not self.enabled or block.training or torch.is_grad_enabled() or block._inference_gemm:
                        return original(value)
                    result=conv(value)
                    if (result.dtype!=torch.float32 or not result.is_cuda or not result.is_contiguous() or
                            tuple(result.shape) not in ((16,256,32),(16,512,16),(16,1024,8),(16,2048,4)) or
                            norm.num_groups!=8 or not norm.affine or activation.inplace or torch.is_autocast_enabled() or
                            any(p.dtype!=torch.float32 or p.device!=result.device or not p.is_contiguous() or
                                p.shape!=(result.shape[1],) for p in (norm.weight,norm.bias))):
                        self.fallbacks[label]+=1
                        return activation(norm(result))
                    self.hits[label]+=1
                    return fused(result,norm.weight,norm.bias,groups=norm.num_groups,eps=norm.eps,fmad=True,affine_mode=2)
                block.forward=forward

    @contextmanager
    def mode(self,enabled):
        previous=self.enabled;self.enabled=enabled
        try:yield self
        finally:self.enabled=previous

    def restore(self):
        for block,original in self.originals:block.forward=original

    def report(self):
        return dict(installed=len(self.originals),hits=dict(self.hits),fallbacks=dict(self.fallbacks),
                    affine_mode=2,reads_live_FP32_affine=True,original_modules_retained=True)


import json
import os

FLAG='UNIFOLM_HEAD_NORM_MISH'


def configured(environment,mode,dataset):
    raw=environment.get(FLAG,'0')
    if raw not in ('0','1'):raise ValueError('Invalid head GN/Mish fusion flag')
    if raw=='0' or dataset!='unitree_z1_dual_arm_stackbox_v2':return False
    if mode!='native':raise ValueError('Head fusion requires native FP32 masters')
    if environment.get('UNIFOLM_SIM_HEAD_CONV1D','native')!='native':
        raise ValueError('Head fusion requires the native Conv1d path')
    return True


def configure_head_fusion(model,*,mode,dataset):
    if not configured(os.environ,mode,dataset):return None
    if hasattr(model,'_head_norm_mish_controller'):raise RuntimeError('Head fusion already installed')
    controller=Controller(model,enabled=True)
    model._head_norm_mish_controller=controller
    print('HEAD_NORM_MISH_INSTALLED '+json.dumps(controller.report()),flush=True)
    return controller
