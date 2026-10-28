"""Change only qualified temporal IEEE FP32 JIT launch warps, never its body."""
from contextlib import contextmanager
import torch
from unifolm_wma.utils.triton_temporal_attention_fp32 import _attention_fp32 as ORIGINAL_JIT

ORIGINAL_RUN=ORIGINAL_JIT.run
active=None
enabled=False

def measure(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype==torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    exact=torch.equal(actual.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))
    assert exact,dict(max_abs=float((actual-expected).abs().max()))
    return dict(relative_l2=0.,max_abs=0.,signedzero_bits=True)

class Controller:
    def __init__(self,configs):
        self.configs={tuple(row['shape']):row for row in configs}
        assert len(self.configs)==len(configs)==5
        assert all(row['warps']==1 and shape[1:]==(16,row['heads']*64) for shape,row in self.configs.items())
        self.captures=[];self.fallbacks={}

    def old(self,reason,grid,args,kwargs):
        self.fallbacks[reason]=self.fallbacks.get(reason,0)+1
        return ORIGINAL_RUN(*args,grid=grid,warmup=False,**kwargs)

    def launch(self,grid,args,kwargs):
        if not enabled or torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.backends.cuda.matmul.allow_tf32:
            return self.old('disabled',grid,args,kwargs)
        values=dict(zip(ORIGINAL_JIT.arg_names,args));values.update(kwargs)
        if any(k not in values for k in ORIGINAL_JIT.arg_names):return self.old('arguments',grid,args,kwargs)
        q=values['Q'];shape=tuple(q.shape);config=self.configs.get(shape)
        if (config is None or values['HEADS']!=config['heads'] or values['SCALE']!=.125 or
                kwargs.get('num_warps')!=4 or kwargs.get('enable_fp_fusion') is not False):
            return self.old('metadata',grid,args,kwargs)
        if any(not t.is_cuda or t.dtype!=torch.float32 or not t.is_contiguous() or t.device!=q.device or t.shape!=q.shape for t in [values[k] for k in ('Q','K','V','Output')]):
            return self.old('layout',grid,args,kwargs)
        if torch.cuda.is_current_stream_capturing():self.captures.append(dict(shape=list(shape),heads=config['heads'],warps=1))
        return ORIGINAL_RUN(*args,grid=grid,warmup=False,**dict(kwargs,num_warps=1))

    def report(self):
        return dict(captured=len(self.captures),captures=self.captures.copy(),fallbacks=self.fallbacks.copy(),
            original_JIT_object_and_body_retained=True,original_all_input_output_storage=True,
            IEEE_FP32_QK_and_PV_and_softmax=True,only_num_warps_changed=True)

def launched(*args,grid=None,warmup=False,**kwargs):
    if warmup:return ORIGINAL_RUN(*args,grid=grid,warmup=True,**kwargs)
    return active.launch(grid,args,kwargs)

def install(configs):
    global active
    assert active is None and ORIGINAL_JIT.run==ORIGINAL_RUN
    active=Controller(configs);ORIGINAL_JIT.run=launched
    return active

def uninstall():
    global active,enabled
    enabled=False;ORIGINAL_JIT.run=ORIGINAL_RUN;active=None

@contextmanager
def mode(value):
    global enabled
    prior=enabled;enabled=value
    try:yield
    finally:enabled=prior
