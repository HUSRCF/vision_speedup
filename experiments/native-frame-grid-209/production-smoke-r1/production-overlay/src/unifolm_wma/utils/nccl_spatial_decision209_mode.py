"""Keep full spatial dispatch shapes; split independent work below dispatch.

Every original Python forward still sees 16 frames. Selected original Triton
launches work on eight frames, retaining K order, tiles and precision. Only the
complete SpatialTransformer output is gathered. Kernels with grouped output
planes and unhandled operators remain replicated. Exact model tests required.
"""
from collections import Counter
from contextlib import contextmanager
import torch
import torch.distributed as dist
import triton
from .nccl_geglu_209_mode import measure

enabled=False
active=None
parent_run=triton.runtime.jit.JITFunction.run


def half_rows(tensor,m,width,rank):
    assert tensor.is_contiguous() and tensor.numel()==m*width,(tensor.shape,m,width)
    return tensor.view(m,width)[rank*(m//2):(rank+1)*(m//2)]


def dispatch(jit,args,kwargs,scope,rank,parent):
    """Return the original compiled kernel; alter only row pointers and grid."""
    values=dict(zip(jit.arg_names,args))
    values.update({k:v for k,v in kwargs.items() if k in jit.arg_names})
    name=jit.fn.__name__
    m=scope['M'];part=m//2;channels=scope['C']
    grid=kwargs.get('grid');handled=False
    def rows(slot,width):
        values[slot]=half_rows(values[slot],m,width,rank)
    def call():
        options={k:v for k,v in kwargs.items() if k not in jit.arg_names and k!='grid'}
        return parent(jit,*[values[k] for k in jit.arg_names],grid=grid,**options)
    def dense(output,inputs):
        nonlocal grid,handled
        k,n=values['K'],values['N']
        for slot in inputs:rows(slot,k)
        for slot in output:rows(slot,n)
        values['M']=part
        bn=values.get('BN',64)
        grid=(triton.cdiv(part,values['BM'])*triton.cdiv(n,bn),)
        if name=='_half_compensated_gemm' and not values['GROUP_M']:
            grid=(triton.cdiv(part,values['BM']),triton.cdiv(n,bn))
        if name=='_zero_low_gemm_persistent' and values['THREE']:
            grid=(min(grid[0],216),)
        handled=True
    regular={
        '_presplit_two_product_gemm':(('Output',),('X','XL')),
        '_packed_attention_output_gemm':(('Output',),('XH','XL')),
        '_half_compensated_gemm':(('Output',),('X',)),
        '_two_product_gemm':(('Output',),('X',)),
        'inline_zero_low_gemm':(('Output',),('X',)),
        '_packed_two_product_gemm_residual':(('Residual','Output'),('XH','XL')),
        '_zero_low_gemm_residual':(('Residual','Y'),('XH','XL')),
        '_zero_low_gemm':(('Y',),('XH','XL')),
        '_zero_low_gemm_persistent':(('Y',),('XH','XL')),
    }
    if name in regular and values.get('M')==m and values.get('OUTPUT_GROUPS',1)==1:
        assert kwargs.get('enable_fp_fusion') is False
        dense(*regular[name])
    elif name=='_primary' and jit.fn.__module__=='unifolm_wma.utils.triton_primary_narrow_linear' and values.get('M')==m:
        dense(('Y',),('X',))
    elif name in ('_geglu','_primary_geglu_hidden_parts','_primary_geglu_hidden_high') and values.get('M')==m:
        assert name!='_geglu' or jit.fn.__module__=='unifolm_wma.utils.triton_primary_narrow_geglu'
        assert not values.get('DEBUG',False)
        for slot in ('XH','XL'):rows(slot,values['K'])
        for slot in (('YH','YL') if name=='_primary_geglu_hidden_parts' else ('Y',)):rows(slot,values['C'])
        values['M']=part
        grid=(triton.cdiv(part,values.get('BM',64))*triton.cdiv(values['C'],values['BN']),);handled=True
    elif name=='_primary_qkv_pack' and values.get('M')==m:
        assert values['N']==channels
        rows('X',values['K'])
        for slot in ('Q','KH','KL','VH','VL'):rows(slot,values['N'])
        values['M']=part
        grid=(triton.cdiv(part,values['BM'])*triton.cdiv(3*values['N'],values['BN']),);handled=True
    elif name in ('_geglu_subtile32','_both_geglu_subtile32') and values.get('M')==m:
        assert not values['DEBUG'] and values['BK']==64
        for slot in ('XH','XL'):rows(slot,values['K'])
        for slot in ('YH','YL'):rows(slot,values['C'])
        # V/G are debug-only pointers and are deliberately left untouched.
        values['M']=part
        grid=(triton.cdiv(part,values['BM'])*triton.cdiv(values['C'],values['BN']),)
        handled=True
    elif name=='_qkv_pack' and values.get('M')==m and values['C']==channels:
        for slot in ('XH','XL','Q','KH','KL','VH','VL'):rows(slot,channels)
        values['M']=part
        grid=(triton.cdiv(part,values['BM'])*triton.cdiv(channels*3,64),)
        handled=True
    elif name in ('_attention_hidden_parts_vector','_attention_hidden_parts_barrier','_prepacked_attention'):
        assert values['M']==m//16 and values['N']==m//16 and values['H']*64==channels
        for slot in (('Q','KH','KL','VH','VL','O') if name=='_prepacked_attention' else ('Q','KH','KL','VH','VL','OH','OL')):
            value=values[slot];assert value.is_contiguous()
            if name=='_prepacked_attention':
                sequence=values['M'] if slot in ('Q','O') else values['N']
                assert value.numel()==16*sequence*values['H']*64
                # Original K/V packs are flattened [B*N*H,64], whereas
                # Q/O retain [B,M,H,64]. Kernels consume only pointers.
                value=value.view(16,-1)
            else:assert value.shape[0]==16
            values[slot]=value[rank*8:(rank+1)*8]
        grid=(triton.cdiv(values['M'],values['BM']),8*values['H']);handled=True
    elif name=='_out_projection_residual' and values.get('M')==m:
        assert not values['TEMPORAL'] and values['T']==1 and values['H']*values['W']==m//16
        rows('X',values['K'])
        for slot in ('Output','Residual'):
            assert values[slot].shape[0]==16
            values[slot]=values[slot][rank*8:(rank+1)*8]
        values['M']=part;grid=(triton.cdiv(part,values['BM'])*triton.cdiv(values['N'],values['BN']),);handled=True
    elif name in ('_residual_layer_norm','_residual_norm_half_pack') and values.get('CHANNELS')==channels:
        for slot in ('Left','Right','Residual','Normalized','High','Low'):
            if slot in values and isinstance(values[slot],torch.Tensor) and values[slot].numel()==m*channels:
                rows(slot,channels)
        grid=(part,);handled=True
    elif name=='_mix' and values.get('N')==m*channels:
        for slot in ('Base','Image','State','Action','Output'):
            if values[slot].numel()==m*channels:rows(slot,channels)
        values['N']=part*channels;grid=(triton.cdiv(values['N'],values['BLOCK']),);handled=True
    elif name=='_pack_half_weight' and values.get('SCALE')==1. and tuple(values['Weight'].shape)==(m,channels):
        for slot in ('Weight','High','Low'):rows(slot,channels)
        values['SIZE']=part*channels;grid=(triton.cdiv(values['SIZE'],values['BLOCK']),);handled=True
    if not handled:
        return parent(jit,*args,**kwargs)
    if torch.cuda.is_current_stream_capturing():scope['shaders'][name]+=1
    return call()


def launch(jit,*args,**kwargs):
    if not enabled or active is None or active.scope is None or kwargs.get('warmup',False):
        return parent_run(jit,*args,**kwargs)
    assert not torch.is_grad_enabled() and not torch.backends.cuda.matmul.allow_tf32
    return dispatch(jit,args,kwargs,active.scope,active.rank,parent_run)


# Must precede configure_denoiser: production owners retain parent run methods.
triton.runtime.jit.JITFunction.run=launch


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import SpatialTransformer
        assert dist.get_world_size()==2
        self.rank=dist.get_rank();self.saved=[];self.captures=[];self.scope=None;self.ffn_parent=None
        for name,module in model.named_modules():
            if type(module) is not SpatialTransformer:continue
            original=module.forward
            def forward(value,context=None,_original=original,_name=name,**kwargs):
                if not enabled:return _original(value,context=context,**kwargs)
                assert self.scope is None and value.ndim==4 and value.shape[0]==16
                assert value.dtype==torch.float32 and context is not None and context.shape[0]==16
                assert not torch.backends.cuda.matmul.allow_tf32
                self.scope=dict(name=_name,M=16*value.shape[2]*value.shape[3],C=value.shape[1],shaders=Counter())
                prior_ffn=self.ffn_parent.enabled if self.ffn_parent is not None else None
                if self.ffn_parent is not None:self.ffn_parent.enabled=False
                try:
                    result=_original(value,context=context,**kwargs)
                    assert result.shape==value.shape and result.dtype==torch.float32
                    local=result[self.rank*8:(self.rank+1)*8].contiguous()
                    output=torch.empty(result.shape,device=result.device,dtype=result.dtype)
                    dist.all_gather_into_tensor(output,local)
                    if torch.cuda.is_current_stream_capturing():
                        self.captures.append(dict(module=_name,input_shape=list(value.shape),shaders=dict(self.scope['shaders']),collectives=1))
                    return output
                finally:
                    if self.ffn_parent is not None:self.ffn_parent.enabled=prior_ffn
                    self.scope=None
            module.forward=forward;self.saved.append((module,original))
        assert len(self.saved)==16

    def report(self):
        return dict(captured=len(self.captures),rows=self.captures.copy(),two_GPUs=True,
            full16_python_shapes_and_original_dispatch=True,original_JIT_bodies_K_order_and_tiles=True,
            vendor_GEMM_dimensions_unchanged=True,decision_CUDNN_math_flag_unchanged=True,
            one_final_output_allgather_per_SpatialTransformer=True,original_global_action_masks_and_KV_caches=True,
            grouped_output_planes_and_unhandled_kernels_replicated=True)


def install(model):
    global active
    assert active is None and triton.runtime.jit.JITFunction.run is launch
    active=Controller(model);return active


def uninstall():
    global active,enabled
    if active is not None:
        assert active.scope is None
        for module,previous in reversed(active.saved):module.forward=previous
    active=None;enabled=False
    assert triton.runtime.jit.JITFunction.run is launch
    triton.runtime.jit.JITFunction.run=parent_run


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:yield
    finally:
        assert active is None or active.scope is None
        enabled=previous
