"""Numerical diagnostic: dense Half outliers plus balanced INT8 residual.

Calibration is frozen from each owner's first eager invocation. Weight packing
is deliberately repeated inside every replay, so synthetic master edits cannot
leave stale quantized weights. Its cost must remain in whole-model timing.
"""
from contextlib import contextmanager
import torch
from unifolm_wma.utils.ffn1280_zero_residual import _zero_low_gemm_residual as ORIGINAL_JIT
import fp16_outlier_packed_residual209_candidate as candidate

ORIGINAL_RUN=ORIGINAL_JIT.run
enabled=False
active=None


def measure(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype==torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    relative=float((actual-expected).double().norm()/expected.double().norm().clamp_min(1e-30))
    maximum=float((actual-expected).abs().max())
    exact=torch.equal(actual.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))
    passed=relative<=.002 and maximum<=.002*max(1.,float(expected.abs().max()))
    result=dict(relative_l2=relative,max_abs=maximum,signedzero_bits=exact,
        gross_passed=passed,full_video_PSNR_required=True)
    assert passed,result
    return result


class Controller:
    def __init__(self):
        self.owners={};self.captures=[];self.fallbacks={}

    def original(self,reason,grid,args,kwargs):
        self.fallbacks[reason]=self.fallbacks.get(reason,0)+1
        return ORIGINAL_RUN(*args,grid=grid,warmup=False,**kwargs)

    def launch(self,grid,args,kwargs):
        if (not enabled or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32):
            return self.original('disabled',grid,args,kwargs)
        v=dict(zip(ORIGINAL_JIT.arg_names,args));v.update(kwargs)
        if any(n not in v for n in ORIGINAL_JIT.arg_names):
            return self.original('arguments',grid,args,kwargs)
        if ([v[n] for n in ('M','N','K','BM','GROUP','OUTPUT_GROUPS')]!=[2560,1280,5120,64,32,1]
                or kwargs.get('num_stages')!=3 or kwargs.get('num_warps')!=4
                or kwargs.get('enable_fp_fusion') is not False):
            return self.original('geometry_or_schedule',grid,args,kwargs)
        high,low=v['WH'],v['WL'];key=(id(high),id(low))
        owner=self.owners.get(key);capturing=torch.cuda.is_current_stream_capturing()
        if owner is None:
            if capturing:return self.original('cold_capture',grid,args,kwargs)
            xlow=v['XL'].view(v['M'],v['K'])
            xmax=xlow.float().abs().amax(0).clamp_min(1e-30)
            wmax=high.float().abs().amax(0).clamp_min(1e-30)
            channel=(wmax.sqrt()/xmax.sqrt()).clamp(2**-16,2**16).contiguous()
            score=xmax*wmax
            outliers=torch.argsort(score,descending=True,stable=True)[:256].to(torch.int32).contiguous()
            channel[outliers.long()]=0
            owner=dict(high=high,low=low,channel=channel,outliers=outliers)
            self.owners[key]=owner
        assert owner['high'] is high and owner['low'] is low
        xp=candidate.allocate_group_pack(v['XL'].view(v['M'],v['K']),v['K'],256)
        wp=candidate.allocate_group_pack(high,v['K'],256)
        candidate.pack_balanced(high,wp,v['K'],owner['channel'],inverse=True,outliers=owner['outliers'])
        candidate.pack_balanced(v['XL'],xp,v['K'],owner['channel'],outliers=owner['outliers'])
        actual=dict(v,_candidate_bm=64,_candidate_stages=3,_launch_options=kwargs,_outlier_indices=owner['outliers'])
        if capturing:self.captures.append(dict(shape=[v[n] for n in ('M','N','K')],
            candidate_bm=64,bn=128,group_k=5120,stages=3,residual_alpha=.5,
            dynamic_weight_and_input_pack=True,exact_Half_residual_K_channels=256,
            dense_Half_outlier_pack_fused_into_INT8_pack=True))
        return candidate.launch_group(actual,xp,wp,v['Y'],bn=128,gk=v['K'])

    def report(self):
        return dict(captured=len(self.captures),captures=self.captures.copy(),
            calibration_owners=len(self.owners),fallbacks=self.fallbacks.copy(),
            calibration='frozen first eager activation per original Half weight owner',
            original_Half_main_unchanged=True,world_middle_FFN_only=True,
            exact_Half_residual_K_channels=256,dense_Half_outlier_pack_fused_into_INT8_pack=True,
            dynamic_weight_and_input_pack_cost_included=True,full_video_PSNR_unverified=True)


def launched(*args,grid=None,warmup=False,**kwargs):
    if warmup:return ORIGINAL_RUN(*args,grid=grid,warmup=True,**kwargs)
    return active.launch(grid,args,kwargs)


def install(configs=None):
    global active
    assert active is None and ORIGINAL_JIT.run==ORIGINAL_RUN
    active=Controller();ORIGINAL_JIT.run=launched
    return active


def uninstall():
    global active,enabled
    ORIGINAL_JIT.run=ORIGINAL_RUN;active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    prior=enabled;enabled=value
    try:yield
    finally:enabled=prior
