"""Numerical and complete-consumer timing study of vendor GEGLU."""
import json,statistics
from pathlib import Path
import torch,triton
import vendor_geglu_209_candidate as candidate
from unifolm_wma.utils.triton_geglu_subtile32 import _geglu_subtile32
from unifolm_wma.utils.triton_geglu_both_subtile32 import _both_geglu_subtile32
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_zero_weight_low_linear import weight_low_count

def err(actual,reference):
    assert actual.shape==reference.shape and torch.isfinite(actual).all() and torch.isfinite(reference).all()
    delta=actual.double()-reference.double()
    return dict(relative_l2=float(delta.norm()/reference.double().norm().clamp_min(1e-30)),max_abs=float(delta.abs().max()))

def guarded(m,c):
    views=[]
    for dtype,sentinel in ((torch.float16,9216.),(torch.float16,9216.),(torch.float32,12345.),(torch.float32,12345.)):
        p=torch.full((m*c+128,),sentinel,device='cuda',dtype=dtype);views.append(p[64:-64].reshape(m,c))
    return tuple(views)

def guards(values):
    for value in values:
        sentinel=9216. if value.dtype==torch.float16 else 12345.
        assert (value._base[:64]==sentinel).all() and (value._base[-64:]==sentinel).all()

def capture(fn,stream):
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):fn()
    graph.replay();torch.cuda.synchronize()
    return graph

def timing(left,right):
    for i in range(100):(left if i%2==0 else right).replay()
    pairs=[]
    for index in range(6):
        row={}
        for name,graph in ([('baseline',left),('candidate',right)] if index%2==0 else [('candidate',right),('baseline',left)]):
            graph.replay();graph.replay();samples=[]
            for _ in range(7):
                start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2));start.record()
                for _ in range(10):graph.replay()
                end.record();end.synchronize();samples.append(start.elapsed_time(end)/10)
            row[name+'_ms']=statistics.median(samples)
        row['saved_ms']=row['baseline_ms']-row['candidate_ms'];pairs.append(row)
    return pairs

@torch.no_grad()
def main():
    torch.manual_seed(4113);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    shapes=[('primary',40960,1280,320,64,16),('primary',10240,2560,640,64,4),
            ('both',2560,5120,1280,64,16),('both',640,5120,1280,32,16),
            ('primary',19,37,65,32,16),('both',19,37,65,32,16)]
    records=[]
    original_runs=(_geglu_subtile32.run,_both_geglu_subtile32.run)
    for index,(kind,m,c,k,bm,group) in enumerate(shapes):
        for has_bias in (False,True):
            value=torch.randn((m,k),device='cuda')*.3
            master=(torch.randn((2*c,k),device='cuda')*.01).half().float()
            bias=torch.randn(2*c,device='cuda')*.01 if has_bias else None
            inputs=pack_half_weight(value,weight_scale=1.);weights=pack_half_weight(master);count=weight_low_count(weights)
            masters=[value,master]+([] if bias is None else [bias]);saved=[v.clone() for v in masters]
            outputs=[guarded(m,c),guarded(m,c)]
            original=_geglu_subtile32 if kind=='primary' else _both_geglu_subtile32
            rr=torch.linspace(0,m-1,min(m,9),device='cuda').round().long().unique()
            cc=torch.linspace(0,c-1,min(c,9),device='cuda').round().long().unique()
            chunks=[k] if index>=4 else sorted({64,k//4,k})
            for chunk in chunks:
                assert k%chunk==0
                buffers=candidate.allocate(m,c,k,chunk,kind,value.device)
                def baseline(debug=True):
                    return original[(triton.cdiv(m,bm)*triton.cdiv(c,64),)](*inputs,*weights,count,
                        weights[0] if bias is None else bias,*outputs[0],m,c,k,has_bias,bm,64,64,group,False,kind=='primary',debug,
                        num_warps=4,num_stages=2,enable_fp_fusion=False)
                def proposed(debug=True,out=None):
                    return candidate.launch(inputs,weights,count,bias,outputs[1] if out is None else out,buffers,kind,chunk,debug)
                baseline();proposed()
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                graphs=[capture(baseline,stream),capture(proposed,stream)]
                rng=torch.cuda.get_rng_state();dynamic=[]
                for mode in ('initial','general','Half','near_constant','zero','restored'):
                    if mode!='initial':
                        value.add_(.001);master.add_(.00013)
                        if bias is not None:bias.add_(.0001)
                        if mode=='Half':value.copy_(value.half().float());master.copy_(master.half().float())
                        elif mode=='near_constant':value.fill_(1e-7)
                        elif mode=='zero':
                            for v in masters:v.zero_()
                        elif mode=='restored':
                            for v,old in zip(masters,saved):v.copy_(old)
                    pack_half_weight(value,weight_scale=1.,out=inputs);pack_half_weight(master,out=weights);weight_low_count(weights,out=count)
                    for parts in outputs:
                        for v in parts:v.fill_(float('nan'))
                    for v in buffers:v.fill_(float('nan'))
                    for graph in graphs:graph.replay()
                    candidate.verify(buffers)
                    for parts in outputs:
                        assert all(torch.isfinite(v).all() for v in parts);guards(parts)
                    expected_v=value[rr].double()@master[cc].double().T
                    if kind=='primary':expected_g=inputs[0][rr].double()@weights[0][cc+c].double().T/256.
                    else:expected_g=value[rr].double()@master[cc+c].double().T
                    if bias is not None:expected_v+=bias[cc].double();expected_g+=bias[cc+c].double()
                    values={}
                    for label,parts in zip(('baseline','candidate'),outputs):
                        values[label+'_value_FP64']=err(parts[2][rr[:,None],cc[None,:]],expected_v)
                        values[label+'_gate_FP64']=err(parts[3][rr[:,None],cc[None,:]],expected_g)
                    values['hidden_vs_baseline']=err(outputs[1][0].float()+outputs[1][1].float()/2048.,outputs[0][0].float()+outputs[0][1].float()/2048.)
                    values.update(mode=mode,Count=int(count.item()));dynamic.append(values)
                    assert values['Count']>0 if mode in ('general','near_constant') else values['Count']==0
                assert torch.equal(rng,torch.cuda.get_rng_state())
                assert all(torch.equal(v.view(torch.int32),old.view(torch.int32)) for v,old in zip(masters,saved))
                fresh=guarded(m,c);proposed(out=fresh)
                for a,b in zip(fresh,outputs[1]):assert torch.equal(a.view(torch.int16 if a.dtype==torch.float16 else torch.int32),b.view(torch.int16 if b.dtype==torch.float16 else torch.int32))
                guards(fresh);del fresh
                numeric=all(d['candidate_value_FP64']['relative_l2']<2e-6 and d['candidate_gate_FP64']['relative_l2']<1e-5 and d['hidden_vs_baseline']['relative_l2']<2e-5 for d in dynamic)
                for graph in graphs:graph.reset()
                graphs=[capture(lambda:baseline(False),stream),capture(lambda:proposed(False),stream)]
                pairs=timing(*graphs) if index<4 and numeric else []
                candidate.verify(buffers);guards(outputs[1])
                row=dict(kind=kind,geometry=[m,c,k],has_bias=has_bias,chunk=chunk,odd=index>=4,
                    numerical_gate=numeric,dynamic=dynamic,pairs=pairs,all_positive=bool(pairs) and all(p['saved_ms']>0 for p in pairs),
                    median_saved_ms=statistics.median(p['saved_ms'] for p in pairs) if pairs else None,
                    NaN_workspace_and_outputs_overwritten=True,canaries=True,fresh_output_exact=True,RNG_unchanged=True,
                    same_original_input_weight_Count_cache=True,all_consumer_work_timed=True,input_weight_packing_excluded_equally=True,
                    new_numerical_reduction_order=True,no_model_or_video_quality_claim=True)
                records.append(row);print('VENDOR_GEGLU_209_ROW '+json.dumps({a:b for a,b in row.items() if a not in ('dynamic','pairs')}),flush=True)
                for graph in graphs:graph.reset()
                del buffers,graphs
            del outputs,inputs,weights,count,value,master,bias,saved,masters
            torch.cuda.empty_cache()
    assert len(records)==28 and (_geglu_subtile32.run,_both_geglu_subtile32.run)==original_runs
    report=dict(passed=True,records=records,eligible=[r for r in records if r['all_positive'] and r['numerical_gate']],
        baseline='canonical209 original coupled GEGLU, original launch configuration',
        synthetic_operands_actual_shapes=True,no_model_install=True,no_full_video_quality_claim=True)
    Path('vendor-geglu-209-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('VENDOR_GEGLU_209_CHECK passed=true fixtures=28',flush=True)

if __name__=='__main__':main()
