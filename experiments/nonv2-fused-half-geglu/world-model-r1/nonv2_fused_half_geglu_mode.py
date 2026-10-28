"""Only the micro-qualified M40960/C320 Half producer is replaced."""
from contextlib import contextmanager
import os
from pathlib import Path
import torch
from nonv2_fused_half_geglu import fused

enabled=False
active=None


@contextmanager
def candidate_cache():
    # Triton2.3 keeps its selected cache-manager class in module globals even
    # after the env var is removed. Restore both so existing model kernels
    # continue using their original executable warm cache.
    import triton.runtime.cache as cache
    previous_env={n:os.environ.get(n)for n in ('TRITON_CACHE_DIR','TRITON_CACHE_MANAGER')}
    previous_class={n:getattr(cache,n)for n in ('__cache_cls','__cache_cls_nme')}
    assert Path(os.environ['FUSED_GEGLU_CACHE_DIR']).is_dir()
    os.environ['TRITON_CACHE_DIR']=os.environ['FUSED_GEGLU_CACHE_DIR']
    os.environ['TRITON_CACHE_MANAGER']='split_experiment_triton_cache:SplitCache'
    try:yield
    finally:
        for n,v in previous_env.items():
            if v is None:os.environ.pop(n,None)
            else:os.environ[n]=v
        for n,v in previous_class.items():setattr(cache,n,v)


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import GEGLU
        self.saved=[];self.rows=[];self.fallbacks=[]
        for name,module in model.named_modules():
            if type(module)is not GEGLU:continue
            before=module.forward
            def forward(value,_before=before,_module=module,_name=name):
                if not enabled:return _before(value)
                c=value.shape[-1];m=value.numel()//c
                if (m,c)!=(40960,320):return _before(value)
                assert not torch.is_grad_enabled() and not torch.is_autocast_enabled() and not _module.training
                assert _module._inference_fused_geglu and value.is_cuda and value.is_contiguous()
                assert value.dtype==_module.proj.weight.dtype==_module.proj.bias.dtype==torch.float16
                assert tuple(_module.proj.weight.shape)==(2560,320)
                with candidate_cache():result=fused(value,_module.proj.weight,_module.proj.bias,(64,128,32,8,3))
                if torch.cuda.is_current_stream_capturing():
                    self.rows.append(dict(module=_name,input_shape=list(value.shape),M=m,C=c,H=1280,
                        config=[64,128,32,8,3],FP16_projection_and_GELU_rounding_boundaries_preserved=True,
                        original_weight_and_bias_tensors_read_directly=True))
                return result
            module.forward=forward;self.saved.append((module,before))
        assert len(self.saved)==33
    def report(self):
        return dict(captured=len(self.rows),rows=self.rows.copy(),registered_GEGLU_modules=len(self.saved),
                    only_M40960_C320=True,consumer_and_all_other_modules_unchanged=True,
                    no_weight_repacking_or_copy_cache=True,full_video_PSNR_unverified=True)


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
