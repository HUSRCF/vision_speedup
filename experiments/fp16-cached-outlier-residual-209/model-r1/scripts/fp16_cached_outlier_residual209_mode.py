"""Cache the PSNR-qualified residual weight pack by its original FP32 master.

The 'dynamic' mode is the previous quantization algorithm, including its weight
pack inside every graph replay. True uses the same arithmetic with a persistent
weight pack; False uses the original production kernel. Half tensor versions
are deliberately not used: original Triton out= writes do not increment them.
"""
from contextlib import contextmanager
import torch
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from unifolm_wma.utils.ffn1280_zero_residual import _zero_low_gemm_residual as ORIGINAL_JIT
import fp16_outlier_packed_residual209_candidate as candidate

ORIGINAL_RUN=ORIGINAL_JIT.run
ORIGINAL_CAPTURE=DenoiserGraph.capture
enabled=False
active=None


def measure(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype==torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    relative=float((actual-expected).double().norm()/expected.double().norm().clamp_min(1e-30))
    maximum=float((actual-expected).abs().max())
    exact=torch.equal(actual.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))
    result=dict(relative_l2=relative,max_abs=maximum,signedzero_bits=exact,gross_passed=exact,
        reference='previous PSNR-qualified dynamic residual packing')
    assert exact,result
    return result


class PackedWeight:
    def __init__(self,module,name,high,low,channel,outliers):
        self.module=module;self.name=name;self.high=high;self.low=low
        self.channel=channel;self.outliers=outliers;self.weight=module.weight
        self.packed=candidate.allocate_group_pack(high,high.shape[1],256)
        self.signature=None;self.refreshes=0;self.refresh_if_needed()

    def original_cache_signature(self):
        weight=self.module.weight
        assert weight is self.weight and weight.dtype==torch.float32
        signature=(weight.data_ptr(),weight._version,tuple(weight.shape),weight.dtype,weight.device)
        cache=getattr(self.module,'_decision_ffn_residual_weight_cache',None)
        assert cache is not None and cache[0]==signature,('original Half master cache stale',self.name)
        assert cache[1][0] is self.high and cache[1][1] is self.low
        return signature

    def refresh_if_needed(self):
        signature=self.original_cache_signature()
        if signature!=self.signature:
            assert not torch.cuda.is_current_stream_capturing(),('cold weight refresh in capture',self.name)
            candidate.pack_balanced(self.high,self.packed,self.high.shape[1],self.channel,
                inverse=True,outliers=self.outliers)
            self.signature=signature;self.refreshes+=1
        for tensor in (*self.packed,self.high,self.low,self.channel,self.outliers):
            tensor.record_stream(torch.cuda.current_stream(tensor.device))

    def verify(self):
        assert self.original_cache_signature()==self.signature,self.name
        expected=candidate.allocate_group_pack(self.high,self.high.shape[1],256)
        candidate.pack_balanced(self.high,expected,self.high.shape[1],self.channel,
            inverse=True,outliers=self.outliers)
        torch.cuda.synchronize(self.high.device)
        for actual,reference in zip(self.packed,expected):
            assert actual.dtype==reference.dtype and actual.shape==reference.shape
            dtype={torch.int8:torch.int8,torch.float16:torch.int16,torch.float32:torch.int32}[actual.dtype]
            assert torch.equal(actual.view(dtype),reference.view(dtype)),self.name
        return dict(module=self.name,refreshes=self.refreshes,
            original_FP32_master_signature_current=True,all_three_packed_arrays_signedzero_bit_exact=True,
            packed_bytes=sum(t.numel()*t.element_size() for t in self.packed))


class Controller:
    def __init__(self,diffusion):
        self.modules=[] if diffusion is None else [(n,m) for n,m in diffusion.named_modules()
            if isinstance(m,torch.nn.Linear) and tuple(m.weight.shape)==(1280,5120)]
        self.owners={};self.captures=[];self.fallbacks={};self.current_capture=None
        self.static_graphs=0;self.dynamic_graphs=0

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
            matches=[]
            for name,module in self.modules:
                cache=getattr(module,'_decision_ffn_residual_weight_cache',None)
                if cache is not None and cache[1][0] is high and cache[1][1] is low:
                    matches.append((name,module))
            assert len(matches)==1,[n for n,m in matches]
            xlow=v['XL'].view(v['M'],v['K'])
            xmax=xlow.float().abs().amax(0).clamp_min(1e-30)
            wmax=high.float().abs().amax(0).clamp_min(1e-30)
            channel=(wmax.sqrt()/xmax.sqrt()).clamp(2**-16,2**16).contiguous()
            outliers=torch.argsort(xmax*wmax,descending=True,stable=True)[:256].to(torch.int32).contiguous()
            channel[outliers.long()]=0
            name,module=matches[0]
            owner=PackedWeight(module,name,high,low,channel,outliers);self.owners[key]=owner
        assert owner.high is high and owner.low is low
        owner.refresh_if_needed()
        xp=candidate.allocate_group_pack(v['XL'].view(v['M'],v['K']),v['K'],256)
        if enabled=='dynamic':
            wp=candidate.allocate_group_pack(high,v['K'],256)
            candidate.pack_balanced(high,wp,v['K'],owner.channel,inverse=True,outliers=owner.outliers)
        else:
            assert enabled is True
            wp=owner.packed
        candidate.pack_balanced(v['XL'],xp,v['K'],owner.channel,outliers=owner.outliers)
        actual=dict(v,_candidate_bm=64,_candidate_stages=3,_launch_options=kwargs,_outlier_indices=owner.outliers)
        if capturing:
            self.captures.append(dict(shape=[v[n] for n in ('M','N','K')],module=owner.name,
                weight_pack_inside_replay=enabled=='dynamic',exact_Half_residual_K_channels=256))
            if self.current_capture is not None and owner not in self.current_capture:
                self.current_capture.append(owner)
        return candidate.launch_group(actual,xp,wp,v['Y'],bn=128,gk=v['K'])

    def capture(self,wrapper,*args,**kwargs):
        use_static=(enabled is True and not torch.backends.cudnn.allow_tf32
            and not torch.backends.cuda.matmul.allow_tf32)
        use_dynamic=(enabled=='dynamic' and not torch.backends.cudnn.allow_tf32
            and not torch.backends.cuda.matmul.allow_tf32)
        previous=self.current_capture
        if use_static:self.current_capture=[]
        try:
            entry=ORIGINAL_CAPTURE(wrapper,*args,**kwargs)
            owners=tuple(self.current_capture or ()) if use_static else ()
        finally:self.current_capture=previous
        if use_static:
            assert len(owners)==10,len(owners)
            native=entry['numerical_sgemm'];assert native is not None
            original=native.replay
            def replay():
                for owner in owners:owner.refresh_if_needed()
                return original()
            native.replay=replay
            entry['residual_quantized_weight_owners']=owners
            self.static_graphs+=1
        elif use_dynamic:self.dynamic_graphs+=1
        return entry

    def report(self):
        verification=[owner.verify() for owner in self.owners.values()]
        return dict(captured=len(self.captures),captures=self.captures.copy(),
            calibration_owners=len(self.owners),fallbacks=self.fallbacks.copy(),
            static_graphs=self.static_graphs,dynamic_reference_graphs=self.dynamic_graphs,
            calibration='same frozen first eager channel scale and outliers in both modes',
            original_Half_main_unchanged=True,world_middle_FFN_only=True,
            exact_Half_residual_K_channels=256,dynamic_input_pack_inside_replay=True,
            static_weight_pack_by_original_FP32_master=True,
            original_Half_cache_master_signature_verified=True,
            weight_owners=verification,full_video_PSNR_unverified=True)


def launched(*args,grid=None,warmup=False,**kwargs):
    if warmup:return ORIGINAL_RUN(*args,grid=grid,warmup=True,**kwargs)
    return active.launch(grid,args,kwargs)


def captured(wrapper,*args,**kwargs):
    return active.capture(wrapper,*args,**kwargs)


def install(diffusion=None):
    global active
    assert active is None and ORIGINAL_JIT.run==ORIGINAL_RUN and DenoiserGraph.capture==ORIGINAL_CAPTURE
    active=Controller(diffusion);ORIGINAL_JIT.run=launched;DenoiserGraph.capture=captured
    return active


def uninstall():
    global active,enabled
    ORIGINAL_JIT.run=ORIGINAL_RUN;DenoiserGraph.capture=ORIGINAL_CAPTURE;active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    assert value is False or value is True or value=='dynamic'
    prior=enabled;enabled=value
    try:yield
    finally:enabled=prior
