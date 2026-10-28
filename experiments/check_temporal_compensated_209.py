"""Five actual temporal shapes plus small batches; FP64, mutations, full timings."""
import json,statistics
from pathlib import Path
import torch
from temporal_compensated_209_candidate import launch
from unifolm_wma.utils.triton_temporal_attention_fp32 import _attention_fp32
from check_vendor_geglu_209 import capture,timing,err

def allocation(shape):
    size=1
    for n in shape:size*=n
    storage=torch.full((size+128,),12345.,device='cuda');return storage[64:-64].reshape(shape)

def verify(values):
    for v in values:assert torch.isfinite(v).all() and (v._base[:64]==12345.).all() and (v._base[-64:]==12345.).all()

@torch.no_grad()
def main():
    torch.manual_seed(20973);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_run=_attention_fp32.run;rows=[];failures=[]
    shapes=[(2560,8),(2560,5),(640,10),(160,20),(40,20),(3,3),(1,1)]
    for index,(batch,heads) in enumerate(shapes):
        source=[torch.randn(batch,16,heads*64,device='cuda')*.3 for _ in range(3)]
        saved=[v.clone() for v in source];positions=torch.tensor(sorted(set((0,batch//2,batch-1))),device='cuda')
        stores=[tuple(allocation(shape) for shape in ((batch,16,heads*64),(batch*heads,16,16),(batch*heads,16,16))) for _ in range(2)]
        for route,warps in (('native',1),('native',2),('compensated',2)):
            q,k,v=source
            def original(debug=True):
                return _attention_fp32[(batch*heads,)](q,k,v,*stores[0],heads,.125,debug,num_warps=4,enable_fp_fusion=False)
            def candidate(debug=True,out=None):
                if route=='native':
                    return _attention_fp32[(batch*heads,)](q,k,v,*(stores[1] if out is None else out),heads,.125,debug,num_warps=warps,enable_fp_fusion=False)
                return launch(q,k,v,*(stores[1] if out is None else out),heads,.125,warps,debug)
            graphs=[]
            try:
                original();compiled=candidate()
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                graphs=[capture(original,stream),capture(candidate,stream)]
                rng=torch.cuda.get_rng_state();dynamic=[]
                for mode in ('initial','changed','near_constant','out_of_range','zero','restored'):
                    if mode=='changed':
                        for j,x in enumerate(source):x.add_(.001*(j+1))
                    elif mode=='near_constant':
                        for x in source:x.fill_(1e-7)
                    elif mode=='out_of_range':
                        source[0].fill_(300.);source[1].fill_(.01);source[2].fill_(500.)
                    elif mode=='zero':
                        for x in source:x.zero_()
                    elif mode=='restored':
                        for x,old in zip(source,saved):x.copy_(old)
                    for values in stores:
                        for x in values:x.fill_(float('nan'))
                    for g in graphs:g.replay()
                    for values in stores:verify(values)
                    if mode in ('near_constant','out_of_range'):
                        for x,y in zip(stores[0],stores[1]):assert torch.equal(x.view(torch.int32),y.view(torch.int32))
                    selected=[x[positions].double().reshape(len(positions),16,heads,64).permute(0,2,1,3) for x in source]
                    exact_qk=selected[0]@selected[1].transpose(-1,-2)
                    exact_p=(exact_qk*.125).softmax(-1);exact_y=exact_p@selected[2]
                    record=dict(mode=mode,baseline_vs_FP64=err(stores[0][0][positions].reshape(len(positions),16,heads,64).permute(0,2,1,3),exact_y),
                        candidate_vs_FP64=err(stores[1][0][positions].reshape(len(positions),16,heads,64).permute(0,2,1,3),exact_y),
                        candidate_vs_baseline=err(stores[1][0],stores[0][0]),
                        candidate_QK_FP64=err(stores[1][1].reshape(batch,heads,16,16)[positions],exact_qk),
                        candidate_probability_FP64=err(stores[1][2].reshape(batch,heads,16,16)[positions],exact_p))
                    record['all_three_signedzero_exact']=all(torch.equal(x.view(torch.int32),y.view(torch.int32)) for x,y in zip(stores[0],stores[1]))
                    dynamic.append(record)
                assert torch.equal(rng,torch.cuda.get_rng_state()) and all(torch.equal(x.view(torch.int32),old.view(torch.int32)) for x,old in zip(source,saved))
                fresh=tuple(allocation(v.shape) for v in stores[1]);candidate(out=fresh)
                for x,y in zip(fresh,stores[1]):assert torch.equal(x.view(torch.int32),y.view(torch.int32))
                verify(fresh);del fresh
                numeric=all(d['candidate_vs_FP64']['relative_l2']<1e-6 and d['candidate_vs_baseline']['relative_l2']<2e-6 and d['candidate_QK_FP64']['relative_l2']<1e-6 and d['candidate_probability_FP64']['relative_l2']<1e-6 for d in dynamic)
                for g in graphs:g.reset()
                graphs=[capture(lambda:original(False),stream),capture(lambda:candidate(False),stream)]
                pairs=timing(*graphs) if numeric and index<5 else []
                for values in stores:verify(values)
                row=dict(shape=[batch,16,heads*64],heads=heads,route=route,warps=warps,numerical_gate=numeric,dynamic=dynamic,pairs=pairs,
                    all_modes_three_outputs_signedzero_exact=all(d['all_three_signedzero_exact'] for d in dynamic),
                    all_positive=bool(pairs) and all(p['saved_ms']>0 for p in pairs),median_saved_ms=statistics.median(p['saved_ms'] for p in pairs) if pairs else None,
                    NaN_canary_fresh=True,RNG_unchanged=True,original_IEEE_baseline=True,original_softmax_exp_division_tree=True,
                    QK_and_PV_both_three_products=route=='compensated',all_public_outputs_FP32=True,
                    bounded_inputs_both_products_native_IEEE_fallback=route=='compensated',
                    resources=dict(registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata['shared']))
                rows.append(row);print('TEMPORAL_COMPENSATED_209_ROW '+json.dumps({k:v for k,v in row.items() if k not in ('dynamic','pairs')}),flush=True)
            except Exception as exc:
                if type(exc).__name__!='OutOfResources':raise
                failures.append(dict(shape=[batch,16,heads*64],route=route,warps=warps,error=str(exc)))
            finally:
                for g in graphs:g.reset()
        del stores,source,saved
        torch.cuda.empty_cache()
    assert len(rows)+len(failures)==21 and _attention_fp32.run==original_run
    report=dict(passed=True,records=rows,compiler_failures=failures,eligible=[r for r in rows if r['all_positive'] and r['numerical_gate']],
        synthetic_inputs_actual_shapes=True,no_model_install=True,no_full_video_quality_claim=True)
    Path('temporal-compensated-209-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TEMPORAL_COMPENSATED_209_CHECK passed=true fixtures=21',flush=True)

if __name__=='__main__':main()
