"""Candidate: shard independent FFN samples on the original nonV2 FP16 path.

Attention, global masks, norms, original FP16 masters and RNG ordering stay
outside this wrapper. Native GEMM M changes may change rounding: GPU/model and
complete official video quality checks are required before any adoption.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import torch
import torch.distributed as dist
enabled=False
active=None


@contextmanager
def candidate_cache(rank):
    """Use an ordinary private cache only for candidate FFN shape variants."""
    assert 'TRITON_CACHE_MANAGER' not in os.environ
    root=Path(os.environ['NONV2_FFN_CACHE_DIR']);assert root.is_dir()
    directory=root/('rank-'+str(rank));directory.mkdir(mode=0o700,exist_ok=True)
    previous=os.environ.get('TRITON_CACHE_DIR')
    os.environ['TRITON_CACHE_DIR']=str(directory)
    try:yield
    finally:
        if previous is None:os.environ.pop('TRITON_CACHE_DIR',None)
        else:os.environ['TRITON_CACHE_DIR']=previous


def row_inference(value,rank,forward,gather):
    assert rank in (0,1) and value.ndim==3 and value.is_contiguous()
    assert value.shape[0]>0 and value.shape[0]%2==0
    part=value.shape[0]//2;rows=slice(rank*part,(rank+1)*part)
    local=forward(value[rows])
    assert local.shape==(part,*value.shape[1:]) and local.dtype==value.dtype and local.device==value.device
    output=torch.empty_like(value,memory_format=torch.contiguous_format)
    gather(output,local.contiguous())
    return output


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import BasicTransformerBlock,FeedForward
        assert dist.is_initialized() and dist.get_world_size()==2
        self.rank=dist.get_rank();self.saved=[];self.rows=[]
        for name,block in model.named_modules():
            if type(block) is not BasicTransformerBlock:continue
            module=block.ff;assert type(module) is FeedForward
            before=module.forward
            def forward(value,_before=before,_module=module,_name=name):
                if not enabled:return _before(value)
                assert not torch.is_grad_enabled() and not torch.is_autocast_enabled() and not _module.training
                assert value.is_cuda and value.dtype==torch.float16 and all(not m.training for m in _module.modules())
                assert all(p.dtype==torch.float16 and p.device==value.device for p in _module.parameters())
                with candidate_cache(self.rank):
                    result=row_inference(value,self.rank,_before,dist.all_gather_into_tensor)
                if torch.cuda.is_current_stream_capturing():
                    self.rows.append(dict(module=_name,input_shape=list(value.shape),dtype=str(value.dtype),rank=self.rank,collectives=1,
                        local_output_MiB=result.numel()*result.element_size()/2**21,
                        original_FFN_forward_and_master_storage=True,native_GEMM_M_halved=True))
                return result
            module.forward=forward;self.saved.append((module,before))
        assert len(self.saved)==33
    def report(self):
        return dict(captured=len(self.rows),rows=self.rows.copy(),original_nonv2_FP16_math_policy=True,
            norms_attention_masks_and_sampling_unchanged=True,one_output_gather_per_FFN=True,
            original_FFN_versioned_weight_owners_retained=True,
            GEMM_dispatch_and_numerics_require_measurement=True,full_video_PSNR_unverified=True)


def install(model):
    global active
    assert active is None;active=Controller(model);return active


def uninstall():
    global active,enabled
    if active is not None:
        for module,forward in reversed(active.saved):module.forward=forward
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:yield
    finally:enabled=previous
