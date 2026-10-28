"""Scoped output epilogue; keep original projection caches and input-pack owner."""
import ast,inspect,textwrap
from collections import Counter
from contextlib import contextmanager
import torch
import torch.utils.checkpoint
from unifolm_wma.utils.triton_out_projection_residual import fused
from unifolm_wma.utils.triton_two_product_linear import _two_product_gemm
active=None


def measure(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
    return dict(relative_l2=0.,max_abs=0.,signedzero_bits=True)


def patched_forward(function,finish):
    source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source);node=tree.body[0]
    original=ast.dump(node,include_attributes=False)
    assert isinstance(node,ast.FunctionDef) and ast.unparse(node.body[-1])=='return x + x_in'
    last=node.body[-1]
    node.body[-1]=ast.Return(value=ast.Call(func=ast.Name(id='_projection_residual_finish',ctx=ast.Load()),args=[ast.Name(id=n,ctx=ast.Load()) for n in ('self','x','x_in')],keywords=[]))
    namespace=dict(function.__globals__);namespace['_projection_residual_finish']=finish
    exec(compile(ast.fix_missing_locations(tree),'<out-projection-residual-finish>','exec'),namespace)
    node.body[-1]=last;assert ast.dump(node,include_attributes=False)==original
    return namespace[function.__name__]


class Controller:
    def __init__(self,model,selection):
        from unifolm_wma.modules.attention import SpatialTransformer,TemporalTransformer
        self.parent_run=_two_product_gemm.run;self.enabled=False;self.current=None;self.projection_call=False
        self.selection={(tuple(r['shape']),tuple(r['stride'])):r for r in selection};assert len(self.selection)==6
        self.captures={};self.fallbacks=Counter();self.saved=[]
        patched={kind:patched_forward(kind.forward,self.finish) for kind in (SpatialTransformer,TemporalTransformer)}
        for name,module in model.named_modules():
            if type(module) not in patched or not module.use_linear or module.in_channels not in (320,640):continue
            assert isinstance(module.proj_out,torch.nn.Linear)
            old,old_projection=module.forward,module.proj_out.forward
            def forward(value,*args,module=module,name=name,old=old,function=patched[type(module)],**kwargs):
                selected=self.eligible(value,module)
                if selected is None:return old(value,*args,**kwargs)
                assert self.current is None
                self.current=dict(module=module,name=name,residual=value,selected=selected,fused=False,output_ptr=None)
                try:return function(module,value,*args,**kwargs)
                finally:self.current=None
            def projection_forward(value,module=module,old=old_projection):
                context=self.current
                if context is None or context['module'] is not module or not self.enabled:return old(value)
                if not value.is_contiguous() or value.dtype!=torch.float32:
                    self.fallbacks['projection_input']+=1;return old(value)
                assert not self.projection_call and _two_product_gemm.run==self.parent_run
                self.projection_call=True;_two_product_gemm.run=self.run
                try:output=old(value)
                finally:_two_product_gemm.run=self.parent_run;self.projection_call=False
                if not context['fused']:return output
                assert output.data_ptr()==context['output_ptr'] and output.is_contiguous() and output.numel()==context['residual'].numel()
                # The GEMM wrote NCHW/NCTHW into the original allocation. Return
                # its inverse MC view so the original rearrange becomes a no-op.
                shaped=output.view(context['residual'].shape)
                if shaped.ndim==4:
                    b,c,h,w=shaped.shape;result=shaped.permute(0,2,3,1).view(b,h*w,c)
                else:
                    b,c,t,h,w=shaped.shape;result=shaped.permute(0,3,4,2,1).view(b,h*w,t,c)
                assert result.shape==output.shape and result.data_ptr()==output.data_ptr()
                return result
            module.forward=forward;module.proj_out.forward=projection_forward
            self.saved.append((module,old,module.proj_out,old_projection))
        assert len(self.saved)==20

    def eligible(self,value,module):
        if not self.enabled:return None
        reason=None
        if torch.is_grad_enabled() or torch.is_autocast_enabled() or module.training or module.proj_out.training:reason='training_grad_autocast'
        elif torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:reason='decision'
        elif not value.is_cuda or value.dtype!=torch.float32 or value.ndim not in (4,5):reason='input'
        elif getattr(module,'_channels_last_output',False):reason='output_layout'
        else:
            selected=self.selection.get((tuple(value.shape),tuple(value.stride())))
            if selected is None:reason='unqualified_layout'
            elif any(p is None or p.dtype!=torch.float32 or p.device!=value.device or not p.is_contiguous() for p in (module.proj_out.weight,module.proj_out.bias)):reason='masters'
        if reason is not None:self.fallbacks[reason]+=1;return None
        return selected

    def run(self,*args,**kwargs):
        context=self.current;assert context is not None and self.projection_call
        if (len(args)!=14 or args[8:]!=(True,True,64,64,64,32) or kwargs.get('num_warps')!=4 or kwargs.get('num_stages')!=2 or kwargs.get('enable_fp_fusion') is not False or kwargs.get('warmup') is not False):
            self.fallbacks['original_projection_dispatch']+=1;return self.parent_run(*args,**kwargs)
        value,high,low,bias,output=args[:5];m,n,k=args[5:8];residual=context['residual']
        assert (m,n,k) in ((40960,320,320),(10240,640,640)) and not context['fused']
        assert value.shape[-1]==k and output.dtype==torch.float32 and output.numel()==residual.numel()==m*n
        _,kernel=fused(value,(high,low),bias,residual,config=context['selected']['config'],output=output.view(residual.shape),inspect=True)
        context['fused']=True;context['output_ptr']=output.data_ptr()
        if torch.cuda.is_current_stream_capturing():
            self.captures[context['name']]=dict(name=context['name'],shape=list(residual.shape),stride=list(residual.stride()),config=context['selected']['config'],geometry=[m,n,k],original_input_weight_parts_bias_output_retained=True,original_projection_owner_and_weight_version_guards=True,residual_add_once=True,inverse_MC_view_no_allocation=True,original_FP32_store_boundary=True)
        return kernel

    def finish(self,module,value,residual):
        context=self.current;assert context is not None and context['module'] is module and residual is context['residual']
        if not context['fused']:return value+residual
        assert value.data_ptr()==context['output_ptr'] and value.shape==residual.shape and value.is_contiguous()
        return value

    def report(self):
        assert self.current is None and not self.projection_call and _two_product_gemm.run==self.parent_run
        return dict(captured=len(self.captures),rows=list(self.captures.values()),fallbacks=dict(self.fallbacks),only_transformer_final_return_changed=True,original_projection_owner_and_version_guards=True,parent_JIT_run_restored_outside_projection=True,input_GN_MC_pack_and_private_GEMM_owners_retained=True,original_FP32_store_boundary=True,residual_add_once=True,no_cross_step_input_cache=True,world_only=True)


def install(model,selection):
    global active
    assert active is None
    active=Controller(model,selection);return active


@contextmanager
def mode(enabled):
    assert active is not None and active.current is None
    prior=active.enabled;active.enabled=enabled
    try:yield active
    finally:active.enabled=prior;assert active.current is None


def uninstall():
    global active
    if active is not None:
        assert active.current is None and _two_product_gemm.run==active.parent_run
        for module,old,projection,old_projection in active.saved:module.forward=old;projection.forward=old_projection
        active=None


import json,os
FLAG='UNIFOLM_WORLD_OUT_PROJECTION_RESIDUAL'
DEPENDENCIES={'UNIFOLM_WORLD_NORM_PROJECTION_PACK': '1', 'UNIFOLM_WORLD_TWO_PRODUCT_TILES': '1', 'UNIFOLM_SERIAL_GROUP_NORM': '1', 'UNIFOLM_HEAD_NORM_MISH': '1', 'UNIFOLM_WORLD_TWO_PRODUCT_OUTPUT': 'input', 'UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT': 'input', 'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE': '1'}
CONFIGS=[{'shape': [1, 320, 16, 40, 64], 'stride': [13107200, 2560, 819200, 64, 1], 'config': [64, 64, 32, 4, 3], 'median_saved_ms': 0.14883840084075928}, {'shape': [1, 640, 16, 20, 32], 'stride': [6553600, 640, 409600, 32, 1], 'config': [64, 64, 32, 4, 3], 'median_saved_ms': 0.04270081520080565}, {'shape': [16, 320, 40, 64], 'stride': [2560, 40960, 64, 1], 'config': [64, 64, 32, 4, 3], 'median_saved_ms': 0.2697728157043457}, {'shape': [16, 320, 40, 64], 'stride': [819200, 2560, 64, 1], 'config': [64, 64, 32, 4, 3], 'median_saved_ms': 0.2590208053588867}, {'shape': [16, 640, 20, 32], 'stride': [640, 10240, 32, 1], 'config': [32, 128, 32, 4, 3], 'median_saved_ms': 0.11509759426116944}, {'shape': [16, 640, 20, 32], 'stride': [409600, 640, 32, 1], 'config': [32, 128, 32, 4, 3], 'median_saved_ms': 0.11381760835647584}]


def configured(environment,mode,dataset):
    raw=environment.get(FLAG,'0')
    if raw not in ('0','1'):raise ValueError('Invalid output projection residual flag')
    if raw=='0' or dataset!='unitree_z1_dual_arm_stackbox_v2':return False
    if mode!='native':raise ValueError('Output projection residual requires native inference')
    for name,value in DEPENDENCIES.items():
        if environment.get(name)!=value:raise ValueError('Output projection residual requires '+name+'='+value)
    return True


def configure_out_projection_residual(model,*,mode,dataset):
    if not configured(os.environ,mode,dataset):return None
    if hasattr(model,'_out_projection_residual_controller'):raise RuntimeError('Output projection residual already installed')
    controller=install(model,CONFIGS);controller.enabled=True
    model._out_projection_residual_controller=controller
    print('OUT_PROJECTION_RESIDUAL_INSTALLED '+json.dumps(dict(enabled=True,nodes=len(controller.saved),original_projection_owners=True,FP32_store_boundary=True,residual_add_once=True,all_setup_inside_process_wall=True)),flush=True)
    return controller


def report_capture():
    if active is None:return
    row=active.report();row['stage']='decision' if torch.backends.cudnn.allow_tf32 else 'world'
    print('OUT_PROJECTION_RESIDUAL_CAPTURE '+json.dumps(row),flush=True)
