"""Split independent spatial-transformer frames, gather only the final output.

This is a numerical candidate: smaller GEMMs can select different arithmetic.
Full video PSNR is mandatory even if the per-forward drift is small.
"""
from contextlib import contextmanager
import torch
import torch.distributed as dist
enabled=False
active=None

def measure(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype==torch.float32
    finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all());assert finite
    exact=torch.equal(actual.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))
    relative=float((actual-expected).norm()/expected.norm().clamp_min(1e-30));maximum=float((actual-expected).abs().max())
    gross=relative<=.02 and maximum<=.02*max(1.,float(expected.abs().max()))
    error=dict(relative_l2=relative,max_abs=maximum,signedzero_bits=exact,finite=finite,gross_passed=gross,full_video_PSNR_required=True)
    assert gross,error
    return error

class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import SpatialTransformer,CrossAttention
        assert dist.get_world_size()==2
        self.rank=dist.get_rank();self.saved=[];self.captures=[];self.modules=[]
        for name,module in model.named_modules():
            if type(module) is not SpatialTransformer:continue
            self.modules.append(module);original=module.forward
            def forward(value,context=None,_original=original,_name=name,**kwargs):
                if not enabled:return _original(value,context=context,**kwargs)
                assert value.shape[0]==16 and context is not None and context.shape[0]==16
                assert not torch.is_grad_enabled() and value.dtype==context.dtype==torch.float32
                assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
                rows=slice(self.rank*8,(self.rank+1)*8);local_kwargs=dict(kwargs)
                if isinstance(local_kwargs.get('mask'),torch.Tensor) and local_kwargs['mask'].shape[0]==16:local_kwargs['mask']=local_kwargs['mask'][rows]
                local=_original(value[rows],context=context[rows],**local_kwargs).contiguous()
                assert local.shape==(8,*value.shape[1:]) and local.dtype==torch.float32
                output=torch.empty(value.shape,device=value.device,dtype=value.dtype)
                dist.all_gather_into_tensor(output,local)
                if torch.cuda.is_current_stream_capturing():self.captures.append(dict(module=_name,input_shape=list(value.shape),frame_start=self.rank*8,frames=8,one_final_output_allgather=True))
                return output
            module.forward=forward;self.saved.append((module,'forward',original))
            for attention in module.modules():
                if not isinstance(attention,CrossAttention):continue
                original_mask=attention._get_attn_mask_aa
                def mask(b,l1,l2,block_size=16,_old=original_mask):
                    if not enabled:return _old(b,l1,l2,block_size)
                    assert b==8 and block_size==16
                    return _old(16,l1,l2,block_size)[self.rank*8:(self.rank+1)*8]
                attention._get_attn_mask_aa=mask;self.saved.append((attention,'_get_attn_mask_aa',original_mask))
                original_projection=attention._project_condition
                def project(name,projection,value,_old=original_projection,_module=attention):
                    if not enabled:return _old(name,projection,value)
                    current=_module._apply_projection(projection,value)
                    target=torch.float32 if _module._inference_attention_self_only else _module._inference_attention_dtype
                    return current.to(target) if current.dtype!=target else current
                attention._project_condition=project;self.saved.append((attention,'_project_condition',original_projection))
                original_prepare=attention.prepare_condition_cache
                def prepare(context,_old=original_prepare):
                    if not enabled:return _old(context)
                    # Candidate K/V are computed from current inputs inside
                    # its graph; the baseline cache remains independent.
                    return None
                attention.prepare_condition_cache=prepare;self.saved.append((attention,'prepare_condition_cache',original_prepare))
        assert len(self.modules)==16

    def report(self):
        return dict(captured=len(self.captures),rows=self.captures.copy(),spatial_modules=len(self.modules),two_GPUs=True,original_FP32_masters=True,original_full16_temporal_and_convolutions=True,action_mask_uses_global_frame_numbers=True,candidate_KV_recomputed_every_step=True,smaller_GEMM_shapes_may_change_arithmetic=True,full_video_PSNR_unverified=True)

def install(model):
    global active
    assert active is None;active=Controller(model);return active

def uninstall():
    global active,enabled
    if active is not None:
        for module,name,method in reversed(active.saved):setattr(module,name,method)
    active=None;enabled=False

@contextmanager
def mode(value):
    global enabled
    prior=enabled;enabled=value
    try:yield
    finally:enabled=prior
