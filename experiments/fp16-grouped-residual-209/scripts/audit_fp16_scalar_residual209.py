"""Fresh in-memory actual operands; export only aggregate accuracy/timing JSON."""
import inspect,json,statistics
from pathlib import Path
import torch
import triton
from triton.runtime.jit import JITFunction
import fp16_grouped_residual209_candidate as candidate


def error(a,b):
    a=a.double();b=b.double();delta=a-b
    return dict(relative_l2=float(torch.linalg.vector_norm(delta)/
        torch.linalg.vector_norm(b).clamp_min(1e-30)),
        max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),
        finite=bool(torch.isfinite(a).all()))


def bits(a,b):
    return torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


def graph(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    result=torch.cuda.CUDAGraph()
    with torch.cuda.graph(result):fn()
    return result


def measure(a,b):
    for _ in range(20):a.replay();b.replay()
    torch.cuda.synchronize();rows=[]
    for pair in range(6):
        measurements={}
        for label,g in ((('baseline',a),('candidate',b)) if pair%2==0 else (('candidate',b),('baseline',a))):
            samples=[]
            for _ in range(7):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):g.replay()
                end.record();end.synchronize();samples.append(start.elapsed_time(end)/10)
            measurements[label]=statistics.median(samples)
        rows.append(dict(**measurements,saved_ms=measurements['baseline']-measurements['candidate']))
    return dict(pairs=rows,median_saved_ms=statistics.median(r['saved_ms'] for r in rows),
        median_speedup=statistics.median(r['baseline']/r['candidate'] for r in rows),
        all_pairs_faster=all(r['saved_ms']>0 for r in rows))


def run(values,jit,directory,label):
    m,n,k=(values[v] for v in ('M','N','K'))
    assert values['OUTPUT_GROUPS']==1
    baseline=torch.empty((m,n),device='cuda',dtype=torch.float32)
    out=torch.empty_like(baseline)
    parameters=inspect.signature(jit.fn).parameters
    def original():
        ordered=[baseline if name=='Y' else values[name] for name in parameters]
        jit[(triton.cdiv(m,values['BM'])*triton.cdiv(n,64),)](*ordered,
            **values['_launch_options'])
    original();torch.cuda.synchronize();reference=baseline.clone()
    assert torch.isfinite(reference).all()
    low_count=int(values['LowCount']);records=[]
    # FP64 oracle evaluates 128 output elements over the complete K reduction.
    ri=torch.linspace(0,m-1,16,device='cuda').long()
    ci=torch.linspace(0,n-1,8,device='cuda').long()
    xd=values['XH'][ri].double()+values['XL'][ri].double()/2048.
    wd=(values['WH'][ci].double()+values['WL'][ci].double()/2048.)/256.
    oracle=xd@wd.T+values['Residual'][ri[:,None],ci[None,:]].double()
    if values['HAS_BIAS']:oracle+=values['Bias'][ci].double()[None,:]
    baseline_oracle=error(reference[ri[:,None],ci[None,:]],oracle)
    base_graph=graph(original)
    xfull=values['XH'].float()+values['XL'].float()/2048.
    underflow=dict(nonzero=int(torch.count_nonzero(xfull)),
        half_zero_from_nonzero=int(torch.count_nonzero((xfull!=0)&(values['XH']==0))),
        min_abs_nonzero=float(xfull.abs().masked_fill(xfull==0,float('inf')).min()),
        max_abs=float(xfull.abs().max()))
    # Powers of two move exponents. Quantify whether range, rather than
    # mantissa rounding, is actually the limiting error on these operands.
    power2=[]
    for scale in (1.,16.,256.,4096.):
        restored=(xfull*scale).half().float()/scale
        power2.append(dict(scale=scale,**error(restored,xfull),
            changed_vs_scale1=int(torch.count_nonzero(restored!=values['XH'].float()))))
    xmax=xfull.abs().amax(0).clamp_min(1e-30)
    wmax=(values['WH'].float()+values['WL'].float()/2048.).abs().amax(0).clamp_min(1e-30)
    del xfull
    configs=[('half_same_correction',1,64,None)]
    if low_count==0:
        configs += [('half_primary_grouped_int8_residual',5,bn,gk)
                    for gk in (256,1024,k) for bn in (64,128)]
    for name,mode,bn,power2_scaling in configs:
        channel=mode==4;half=mode in (3,4)
        # Match the original depth where the larger N tile's fallback shared
        # memory fits. N128 remains an explicitly recorded depth2 candidate.
        values['_candidate_stages']=values['_launch_options'].get('num_stages',2) if bn==64 else 2
        if mode==5:
            gk=power2_scaling
            xp=candidate.allocate_group_pack(values['XL'],gk)
            wp=candidate.allocate_group_pack(values['WH'],gk)
        else:
            xp=candidate.allocate_pack(values['XH'] if half else values['XL'],half=half)
            wp=candidate.allocate_pack(values['WH'],half=half)
        if channel:
            alpha=power2_scaling
            channel_scale=(wmax.pow(1-alpha)/xmax.pow(alpha)).clamp(2**-8,2**8).contiguous()
        loww=values['WL'] if half else None
        # Static preparation cost is recorded separately; dynamic pack is
        # included in every graph replay, before the GEMM.
        sw=torch.cuda.Event(enable_timing=True);ew=torch.cuda.Event(enable_timing=True)
        sw.record()
        if mode==5:candidate.pack_group(values['WH'],wp,gk)
        elif channel:candidate.pack_channels(values['WH'],values['WL'],channel_scale,wp[0],inverse=True)
        else:candidate.pack(values['WH'],wp,half=half,power2=bool(power2_scaling),low=loww)
        ew.record();ew.synchronize();static_pack_ms=sw.elapsed_time(ew)
        def call():
            if mode==5:candidate.pack_group(values['XL'],xp,gk)
            elif channel:
                candidate.pack_channels(values['XH'],values['XL'],channel_scale,xp[0])
            elif mode in (2,3):
                candidate.pack(values['XH'] if half else values['XL'],xp,half=half,
                    power2=bool(power2_scaling),low=values['XL'] if half else None)
            actual=dict(values,XH=xp[0],WH=wp[0]) if channel else values
            if mode==5:candidate.launch_group(actual,xp,wp,out,bn=bn,gk=gk)
            else:candidate.launch(actual,xp,wp,out,mode=0 if channel else mode,bn=bn)
        out.fill_(float('nan'));call();torch.cuda.synchronize()
        initial=error(out,reference)
        sampled=error(out[ri[:,None],ci[None,:]],oracle)
        delta=reference-values['Residual']
        if values['HAS_BIAS']:delta=delta-values['Bias'][None,:]
        residual_free_relative=float(torch.linalg.vector_norm((out-reference).double())/
            torch.linalg.vector_norm(delta.double()).clamp_min(1e-30))
        if mode==1:assert bits(out,reference),'Original K64 compensation must match bitwise'
        eligible=bool(initial['finite'] and residual_free_relative<=6e-6 and
                      sampled['relative_l2']<=1e-5)
        if mode in (0,3,4):assert low_count==0
        candidate_graph=graph(call)
        # Change only fresh clones, never the live model/master inputs.
        saved=values['XL'].clone();values['XL'].mul_(.75)
        original();call();dynamic_error=error(out,baseline)
        assert not bits(baseline,reference),'Input sensitivity witness failed'
        values['XL'].copy_(saved);del saved
        if mode in (2,5):
            # General nonzero weight low bits use the original Half path.
            assert low_count==0
            saved=values['WL'].clone();values['WL'].fill_(.125);values['LowCount'].fill_(values['WL'].numel())
            original();call();assert bits(out,baseline),'Nonzero-WL fallback must be exact'
            values['WL'].copy_(saved);values['LowCount'].zero_();del saved
        original();call();assert bits(baseline,reference)
        assert torch.isfinite(out).all()
        measured=measure(base_graph,candidate_graph)
        candidate_graph.replay();torch.cuda.synchronize()
        replay_error=error(out,reference)
        assert replay_error==initial
        record=dict(name=name,mode=mode,bn=bn,num_stages=values['_candidate_stages'],
            channel_alpha=power2_scaling if channel else None,
            channel_calibration_scope='this first-world sample only; no full-step calibration' if channel else None,
            static_weight_pack_ms_includes_first_JIT=static_pack_ms,
            error=initial,relative_l2_without_residual=residual_free_relative,
            fp64_sample_error=sampled,dynamic_input_error=dynamic_error,
            numerical_eligible_for_model_only=eligible,
            nonzero_weight_low_fallback_bitwise=mode in (2,5),
            residual_scale_group_k=gk if mode==5 else None,
            dynamic_pack_cost_included=True,**measured)
        records.append(record)
        print('FP16_SCALAR_RESULT '+json.dumps(dict(label=label,shape=[m,n,k],**record)),flush=True)
        candidate_graph.reset()
        (directory/'fp16-scalar-progress.json').write_text(json.dumps(records,indent=2)+'\n')
    base_graph.reset()
    return dict(label=label,shape=[m,n,k],weight_low_nonzero=low_count,
        original_launch_options=values['_launch_options'],
        baseline_fp64_sample_error=baseline_oracle,input_range=underflow,
        power2_half_reconstruction=power2,records=records,
        no_operand_tensors_serialized=True,no_production_install=True)


class Audit:
    def __init__(self,model):
        self.parent=JITFunction.run;self.values={};self.collect=False
        owner=self
        def launch(jit,*args,**kwargs):
            if owner.collect and jit.fn.__name__=='_zero_low_gemm_residual':
                bound=dict(zip(jit.arg_names,args))
                bound.update({n:v for n,v in kwargs.items() if n in jit.arg_names})
                shape=tuple(bound[n] for n in ('M','N','K'))
                if shape not in owner.values and bound['OUTPUT_GROUPS']==1:
                    cloned={n:(v.detach().clone() if isinstance(v,torch.Tensor) else v)
                            for n,v in bound.items() if n!='Y'}
                    cloned['_launch_options']={n:kwargs[n] for n in
                        ('num_warps','num_stages','enable_fp_fusion') if n in kwargs}
                    owner.values[shape]=(jit,cloned)
            return owner.parent(jit,*args,**kwargs)
        self.launch=launch;JITFunction.run=launch

    def report(self,wrapper,model,output,values,options):
        output=Path(output);self.collect=True
        try:wrapper.original(*values,**options)
        finally:self.collect=False
        torch.cuda.synchronize()
        assert set(self.values)=={(640,1280,5120),(2560,1280,5120),(40960,320,1280)},sorted(self.values)
        records=[]
        for shape,(jit,operands) in sorted(self.values.items()):
            records.append(run(operands,jit,output,str(shape)))
        report=dict(passed=True,actual_fresh_model_operands=True,
            records=records,no_official_score=True,
            only_aggregate_statistics_exported=True,
            full_video_PSNR_required=True)
        (output/'fp16-scalar-residual-report.json').write_text(json.dumps(report,indent=2)+'\n')
        self.values.clear()

    def close(self):
        assert JITFunction.run==self.launch
        JITFunction.run=self.parent
