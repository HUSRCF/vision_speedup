"""Reuse qualified native FFN row grids and the original GEGLU CUBIN locally."""
import hashlib
import torch
import triton
from unifolm_wma.utils.triton_geglu import _geglu
import nonv2_native_ffn_grid59_mode as adopted

parent_run=triton.runtime.jit.JITFunction.run
active=None


def local_arguments(projected,output,n,channels,block,rank):
    assert rank in(0,1)and projected.is_contiguous()and output.is_contiguous()
    assert projected.dtype==output.dtype==torch.float16 and projected.numel()==2*n and output.numel()==n
    assert n%2==0 and n//2%channels==0 and n//2%block==0
    first=projected.view(-1)[rank*n:(rank+1)*n]
    second=output.view(-1)[rank*(n//2):(rank+1)*(n//2)]
    assert first.numel()==n and second.numel()==n//2
    return first,second


def cubin_sha(kernel):return hashlib.sha256(kernel.asm['cubin']).hexdigest()


def launch(jit,*args,**kwargs):
    if jit is not _geglu or active is None or kwargs.get('warmup',False):return parent_run(jit,*args,**kwargs)
    projected,output,n,channels,block=args;key=(n,channels,block)
    if not adopted.enabled or active.scope is None:
        result=parent_run(jit,*args,**kwargs);digest=cubin_sha(result)
        previous=active.original_cubins.setdefault(key,digest);assert previous==digest
        return result
    assert len(args)==5 and tuple(kwargs['grid'])==(triton.cdiv(n,block),)
    assert projected.is_cuda and not torch.is_grad_enabled()
    if key not in active.original_cubins:
        assert not torch.cuda.is_current_stream_capturing()
        active.original_cubins[key]=cubin_sha(parent_run(jit,*args,**kwargs))
    local_projected,local_output=local_arguments(projected,output,n,channels,block,active.rank)
    options=dict(kwargs,grid=(n//2//block,))
    result=parent_run(jit,local_projected,local_output,n,channels,block,**options)
    digest=cubin_sha(result);assert digest==active.original_cubins[key]
    if torch.cuda.is_current_stream_capturing():active.geglu_rows.append(dict(module=active.scope['name'],N=n,channels=channels,block=block,original_CUBIN_SHA256=digest,original_N_constexpr_and_rounding=True,owned_half_tiles_without_tail_mask_changes=True))
    return result


class Controller(adopted.Controller):
    def __init__(self,model):
        self.original_cubins={};self.geglu_rows=[];super().__init__(model)
    def report(self):
        report=super().report()
        report['GEGLU']=dict(captured=len(self.geglu_rows),rows=self.geglu_rows.copy(),no_new_JIT_kernel_or_arithmetic=True,original_compiled_CUBIN_matched=True,full_N_constexpr_and_GELU_Half_rounding_preserved=True)
        return report


def install(model):
    global active
    assert active is None and adopted.active is None and triton.runtime.jit.JITFunction.run is parent_run
    active=Controller(model);adopted.active=active;triton.runtime.jit.JITFunction.run=launch;return active


def uninstall():
    global active
    assert triton.runtime.jit.JITFunction.run is launch;triton.runtime.jit.JITFunction.run=parent_run
    adopted.uninstall();active=None


mode=adopted.mode
