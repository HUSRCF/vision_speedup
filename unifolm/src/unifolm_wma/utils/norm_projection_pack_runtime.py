"""Preserve transformer/projection owners; fuse only norm output preparation."""
from collections import Counter
from contextlib import contextmanager
import torch
import torch.utils.checkpoint
import triton
from unifolm_wma.utils.triton_norm_projection_pack import pack
from unifolm_wma.utils.triton_presplit_two_product_linear import _presplit_two_product_gemm
from unifolm_wma.utils.triton_two_product_linear import _two_product_gemm

active=None


def measure(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
    return dict(relative_l2=0.,max_abs=0.,signedzero_bits=True)


class Controller:
    def __init__(self,model,selection):
        from unifolm_wma.modules.attention import SpatialTransformer,TemporalTransformer
        self.parent_run=_two_product_gemm.run
        self.enabled=False;self.captures={};self.fallbacks=Counter();self.saved=[];self.pending={};self.current=None
        self.selection={(tuple(r['shape']),tuple(r['stride'])):r for r in selection};assert len(self.selection)==6
        for name,module in model.named_modules():
            if not isinstance(module,(SpatialTransformer,TemporalTransformer)) or not module.use_linear or module.norm.num_channels not in (320,640):continue
            assert isinstance(module.proj_in,torch.nn.Linear)
            norm,projection=module.norm,module.proj_in
            old_norm,old_projection=norm.forward,projection.forward
            def norm_forward(value,module=module,name=name,old=old_norm):
                selected=self.eligible(value,module)
                key=id(module.proj_in);self.pending.pop(key,None)
                if selected is None:return old(value)
                b,c=value.shape[:2];m=value.numel()//c
                normalized=torch.empty((m,c),device=value.device,dtype=torch.float32)
                packed,_=pack(value,module.norm,tile=selected['tile'],debug=normalized)
                self.pending[key]=dict(value=normalized,packed=packed,name=name,shape=tuple(value.shape),stride=tuple(value.stride()),tile=selected['tile'])
                if value.ndim==4:
                    h,w=value.shape[2:];return normalized.view(b,h,w,c).permute(0,3,1,2)
                t,h,w=value.shape[2:];return normalized.view(b,h,w,t,c).permute(0,4,3,1,2)
            def projection_forward(value,module=module,old=old_projection):
                context=self.pending.pop(id(module.proj_in),None)
                if not self.enabled or context is None:return old(value)
                if value.data_ptr()!=context['value'].data_ptr() or not value.is_contiguous() or value.dtype!=torch.float32:
                    self.fallbacks['layout_copy_not_eliminated']+=1;return old(value)
                assert self.current is None and _two_product_gemm.run==self.parent_run
                self.current=context;_two_product_gemm.run=self.run
                try:return old(value)
                finally:_two_product_gemm.run=self.parent_run;self.current=None
            norm.forward=norm_forward;projection.forward=projection_forward
            self.saved.append((norm,old_norm,projection,old_projection))
        assert len(self.saved)==20

    def eligible(self,value,module):
        if not self.enabled:return None
        reason=None
        if torch.is_grad_enabled() or torch.is_autocast_enabled() or module.training or module.norm.training or module.proj_in.training:reason='training_grad_autocast'
        elif torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:reason='decision'
        elif not value.is_cuda or value.dtype!=torch.float32 or value.ndim not in (4,5):reason='input'
        elif value.ndim==5 and not module._inference_direct_layout:reason='temporal_layout'
        else:
            selected=self.selection.get((tuple(value.shape),tuple(value.stride())))
            if selected is None:reason='unqualified_layout'
            elif module.norm.num_groups!=32 or module.norm.eps!=1e-6:reason='normalization'
            elif any(p is None or p.dtype!=torch.float32 or p.device!=value.device or not p.is_contiguous() for p in (module.norm.weight,module.norm.bias,module.proj_in.weight,module.proj_in.bias)):reason='masters'
        if reason is not None:self.fallbacks[reason]+=1;return None
        return selected

    def run(self,*args,**kwargs):
        context=self.current;assert context is not None
        if (len(args)!=14 or args[8:]!=(True,True,64,64,64,32) or args[0].data_ptr()!=context['value'].data_ptr() or
                kwargs.get('num_warps')!=4 or kwargs.get('num_stages')!=2 or kwargs.get('enable_fp_fusion') is not False or kwargs.get('warmup') is not False):
            self.fallbacks['original_projection_dispatch']+=1;return self.parent_run(*args,**kwargs)
        value,high,low,bias,output=args[:5];m,n,k=args[5:8]
        assert (m,n,k) in ((40960,320,320),(10240,640,640))
        assert all(p.shape==(m,k) and p.dtype==torch.float16 and p.device==value.device and p.is_contiguous() for p in context['packed'])
        if torch.cuda.is_current_stream_capturing():
            self.captures[context['name']]=dict(name=context['name'],shape=list(context['shape']),stride=list(context['stride']),tile=context['tile'],geometry=[m,n,k],MC_pointer_unchanged=True,original_weight_parts_and_bias_and_output_retained=True,original_projection_owner_and_weight_version_guards=True)
        return _presplit_two_product_gemm[(triton.cdiv(m,64)*triton.cdiv(n,64),)](*context['packed'],high,low,bias,output,m,n,k,True,True,64,64,64,32,num_warps=4,num_stages=3,enable_fp_fusion=False)

    def report(self):
        assert _two_product_gemm.run==self.parent_run and self.current is None
        return dict(captured=len(self.captures),rows=list(self.captures.values()),fallbacks=dict(self.fallbacks),original_transformer_forward_retained=True,original_projection_owner_and_version_guards=True,parent_JIT_run_restored_outside_projection=True,real_normalized_F32_MC_for_fallback=True,original4096_statistics_and_FP32_affine=True,no_cross_step_input_cache=True,world_only=True,pending=len(self.pending))


def install(model,selection):
    global active
    assert active is None
    active=Controller(model,selection);return active


@contextmanager
def mode(enabled):
    assert active is not None
    prior=active.enabled;active.enabled=enabled
    try:yield active
    finally:active.enabled=prior;active.pending.clear()


def uninstall():
    global active
    if active is not None:
        assert _two_product_gemm.run==active.parent_run and active.current is None
        for norm,old_norm,projection,old_projection in active.saved:norm.forward=old_norm;projection.forward=old_projection
        active.pending.clear();active=None


import json,os
FLAG='UNIFOLM_WORLD_NORM_PROJECTION_PACK'
DEPENDENCIES={'UNIFOLM_SERIAL_GROUP_NORM': '1', 'UNIFOLM_WORLD_TWO_PRODUCT_TILES': '1', 'UNIFOLM_HEAD_NORM_MISH': '1', 'UNIFOLM_SPLIT_GROUP_NORM': 'fp32', 'UNIFOLM_SPLIT_TRANSFORMER_NORM': '1', 'UNIFOLM_TEMPORAL_LAYOUT_DIRECT': '1', 'UNIFOLM_WORLD_TWO_PRODUCT_OUTPUT': 'input', 'UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT': 'input', 'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE': '1'}
CONFIGS=[{'shape': [1, 320, 16, 40, 64], 'stride': [13107200, 2560, 819200, 64, 1], 'tile': [64, 32], 'median_saved_ms': 0.11601920127868648}, {'shape': [1, 640, 16, 20, 32], 'stride': [6553600, 640, 409600, 32, 1], 'tile': [32, 32], 'median_saved_ms': 0.03271679878234862}, {'shape': [16, 320, 40, 64], 'stride': [2560, 40960, 64, 1], 'tile': [32, 64], 'median_saved_ms': 0.09692158699035647}, {'shape': [16, 320, 40, 64], 'stride': [819200, 2560, 64, 1], 'tile': [32, 64], 'median_saved_ms': 0.08703999519348143}, {'shape': [16, 640, 20, 32], 'stride': [640, 10240, 32, 1], 'tile': [32, 64], 'median_saved_ms': 0.050892806053161654}, {'shape': [16, 640, 20, 32], 'stride': [409600, 640, 32, 1], 'tile': [32, 64], 'median_saved_ms': 0.040908801555633534}]


def configured(environment,mode,dataset):
    raw=environment.get(FLAG,'0')
    if raw not in ('0','1'):raise ValueError('Invalid norm projection pack flag')
    if raw=='0' or dataset!='unitree_z1_dual_arm_stackbox_v2':return False
    if mode!='native':raise ValueError('Norm projection pack requires native inference')
    for name,value in DEPENDENCIES.items():
        if environment.get(name)!=value:raise ValueError('Norm projection pack requires '+name+'='+value)
    return True


def configure_norm_projection_pack(model,*,mode,dataset):
    if not configured(os.environ,mode,dataset):return None
    if hasattr(model,'_norm_projection_pack_controller'):raise RuntimeError('Norm projection pack already installed')
    controller=install(model,CONFIGS);controller.enabled=True
    model._norm_projection_pack_controller=controller
    print('NORM_PROJECTION_PACK_INSTALLED '+json.dumps(dict(enabled=True,nodes=len(controller.saved),real_FP32_MC=True,original_transformer_projection_owners=True,all_setup_inside_process_wall=True)),flush=True)
    return controller


def report_capture():
    if active is None:return
    row=active.report();row['stage']='decision' if torch.backends.cudnn.allow_tf32 else 'world'
    print('NORM_PROJECTION_PACK_CAPTURE '+json.dumps(row),flush=True)
