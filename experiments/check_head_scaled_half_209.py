"""Real-head numerical/speed study against cached native cuDNN graphs."""
import hashlib,importlib.util,json,statistics,time,traceback
from pathlib import Path
import torch,triton
from torch.nn import functional as F
from unifolm_wma.utils.native_graph_weight_cache_probe import TRANSPOSE,GraphAPI,NativeWeightCache
from unifolm_wma.utils.world_head_filter_cache import plan_for
import head_scaled_half_209_candidate as candidate

def rna(value):return ((value.contiguous().view(torch.int32)+0x1000)&-8192).view(torch.float32)

def error(actual,reference):
    assert actual.shape==reference.shape and torch.isfinite(actual).all() and torch.isfinite(reference).all()
    delta=actual.double()-reference.double()
    return dict(relative_l2=float(delta.norm()/reference.double().norm().clamp_min(1e-30)),max_abs=float(delta.abs().max()))

def subset(value,positions,channels):
    length=value.shape[2]
    return value[positions[:,None]//length,channels[None,:],positions[:,None]%length]

def oracle(x,weight,bias,metadata,quantized=True):
    batch,ci,length=x.shape;co,_,ks=weight.shape
    stride,pad,dil=metadata['stride'][0],metadata['padding'][0],metadata['dilation'][0]
    ol=(length+2*pad-dil*(ks-1)-1)//stride+1
    rows=torch.linspace(0,batch*ol-1,min(9,batch*ol),device=x.device).round().long().unique()
    cols=torch.linspace(0,co-1,min(9,co),device=x.device).round().long().unique()
    xx=F.pad((rna(x) if quantized else x).double(),(pad,pad))
    chan=torch.arange(ci,device=x.device);tap=torch.arange(ks,device=x.device)
    patch=xx[(rows//ol)[:,None,None],chan[None,:,None],((rows%ol)*stride)[:,None,None]+tap[None,None,:]*dil]
    ww=rna(weight) if quantized else weight
    result=patch.reshape(rows.numel(),ci*ks)@ww.double()[cols].reshape(cols.numel(),ci*ks).T
    if bias is not None:result+=bias.double()[cols]
    return result,rows,cols

def capture(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph,stream=stream):result=fn()
    graph.replay();torch.cuda.synchronize()
    return graph,result

def native_cache(graph,weight,extension):
    api=GraphAPI();raw,_=extension.get_handles(graph)
    inventory=api.read(raw,dict(weight=weight));nodes={n['node']:n for n in inventory['nodes']};outgoing={}
    for a,b in inventory['edges']:outgoing.setdefault(a,[]).append(b)
    transforms=[n for n in nodes.values() if n.get('name')==TRANSPOSE and n.get('matched_tensors')==['weight']]
    summary=[dict(name=n.get('name'),type=n['type'],matched=n.get('matched_tensors'),argument_sizes=[a['size'] for a in n.get('arguments',[])]) for n in nodes.values()]
    if len(transforms)!=1:raise ValueError(json.dumps(dict(transform_count=len(transforms),kernels=summary)))
    try:plan=plan_for(nodes,outgoing,transforms[0],weight)
    except AssertionError as exc:raise ValueError(json.dumps(dict(plan_error=traceback.format_exc(),kernels=summary))) from exc
    cached=NativeWeightCache(graph,weight,extension,api,validated_plan=plan)
    cached.verify_cache();cached.set_cached(True)
    return cached

def timings(left,right):
    for i in range(100):(left if i%2==0 else right).replay()
    pairs=[]
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for index in range(6):
        pair={}
        for label,graph in ([('baseline',left),('candidate',right)] if index%2==0 else [('candidate',right),('baseline',left)]):
            graph.replay();graph.replay();samples=[]
            for _ in range(7):
                start.record()
                for _ in range(10):graph.replay()
                end.record();end.synchronize();samples.append(start.elapsed_time(end)/10)
            pair[label+'_ms']=statistics.median(samples)
        pair['saved_ms']=pair['baseline_ms']-pair['candidate_ms'];pairs.append(pair)
    return pairs

@torch.no_grad()
def main():
    config=json.loads(Path('head-scaled-half-209-config.json').read_text())
    path=Path(config['input_source']);assert hashlib.sha256(path.read_bytes()).hexdigest()==config['input_sha256']
    inputs=torch.load(path,map_location='cpu');checkpoint=torch.load(config['checkpoint'],map_location='cpu',mmap=True)
    masters=checkpoint.get('state_dict',checkpoint)
    so=Path(__import__('unifolm_wma.utils.native_graph_weight_cache',fromlist=['x']).__file__).parent/'native-graph-handles/team07_cuda_graph_handles.so'
    assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so);extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    torch.manual_seed(20901);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    rows=[];unsupported=[];failures=[]
    cases=[]
    for name in config['modules']:
        record=inputs[name];meta=record['metadata'];assert meta['kind']=='Conv1d' and meta['groups']==1
        assert meta['weight_name'] in masters
        cases.append((name,record['input'],masters[meta['weight_name']],masters.get(meta['weight_name'].removesuffix('weight')+'bias'),meta,False))
    seed_x=torch.arange(2*17*7).reshape(2,17,7).float().sin()*.3
    seed_w=torch.arange(19*17*3).reshape(19,17,3).float().cos()*.01
    for has_bias in (False,True):
        meta=dict(kind='Conv1d',groups=1,stride=[1],padding=[1],dilation=[1],cudnn_benchmark=False)
        cases.append(('odd_bias_'+str(has_bias),seed_x,seed_w,torch.zeros(19) if has_bias else None,meta,True))
    for name,cpu_x,cpu_weight,cpu_bias,metadata,odd in cases:
        x=cpu_x.to('cuda',torch.float32).contiguous();weight=cpu_weight.to('cuda',torch.float32).contiguous()
        bias=cpu_bias.to('cuda',torch.float32).contiguous() if cpu_bias is not None else None
        torch.backends.cudnn.benchmark=metadata['cudnn_benchmark']
        native=lambda:F.conv1d(x,weight,bias,stride=metadata['stride'][0],padding=metadata['padding'][0],dilation=metadata['dilation'][0],groups=1)
        baseline,y0=capture(native);reference=y0.clone();cached=None
        if not odd:
            try:cached=native_cache(baseline,weight,extension)
            except (AssertionError,ValueError) as exc:
                unsupported.append(dict(module=name,reason=str(exc),excluded_from_all_speed_claims=True));baseline.reset();continue
            baseline.replay();assert torch.equal(reference.view(torch.int32),y0.view(torch.int32))
        half=torch.empty_like(weight,dtype=torch.float16)
        def refresh():
            if cached is not None:cached.refresh_if_needed()
            candidate.pack[(triton.cdiv(weight.numel(),1024),)](weight,half,weight.numel(),1024,num_warps=4,enable_fp_fusion=False)
        refresh();torch.cuda.synchronize()
        input_rep=bool(torch.equal(((rna(x)*256).half().float()/256).view(torch.int32),rna(x).view(torch.int32)))
        weight_rep=bool(torch.equal((half.float()/256).view(torch.int32),rna(weight).view(torch.int32)))
        expected,pos,ch=oracle(x,weight,bias,metadata);native_error=error(subset(y0,pos,ch),expected)
        saved_x=x.clone();saved_weight=weight.clone();saved_bias=bias.clone() if bias is not None else None
        settings=config['tiles'] if not odd else config['tiles'][:2]
        for tile in settings:
            backing=torch.full((y0.numel()+128,),90909.,device='cuda');out=backing[64:-64].reshape_as(y0);graph=None
            try:
                graph,kernel=capture(lambda:candidate.launch(x,half,bias,out,metadata,tile))
            except Exception as exc:
                if type(exc).__name__!='OutOfResources':raise
                failures.append(dict(module=name,config=tile,error=str(exc)));continue
            rng=torch.cuda.get_rng_state();dynamic=[]
            def validate(label):
                out.fill_(float('nan'));baseline.replay();graph.replay()
                if cached is not None:
                    assert cached.version==weight._version
                    cached.verify_canaries()
                reference,pp,cc=oracle(x,weight,bias,metadata)
                assert torch.isfinite(out).all() and torch.isfinite(y0).all()
                ne=error(subset(y0,pp,cc),reference);ce=error(subset(out,pp,cc),reference)
                native_raw_error=None
                if odd:
                    raw_reference,raw_pp,raw_cc=oracle(x,weight,bias,metadata,quantized=False)
                    native_raw_error=error(subset(y0,raw_pp,raw_cc),raw_reference)
                difference=error(out,y0)
                assert (backing[:64]==90909.).all() and (backing[-64:]==90909.).all()
                dynamic.append(dict(mode=label,native_FP64=ne,native_raw_FP64=native_raw_error,candidate_FP64=ce,vs_native=difference))
            validate('initial')
            fresh=torch.empty_like(out);candidate.launch(x,half,bias,fresh,metadata,tile)
            assert torch.equal(fresh.view(torch.int32),out.view(torch.int32));del fresh
            for mode in ('general','Half','zero','restored'):
                x.add_(.0001);weight.add_(.00013)
                if bias is not None:bias.add_(.0001)
                if mode=='Half':weight.copy_(weight.half().float())
                elif mode=='zero':weight.zero_();x.zero_()
                else:
                    if mode=='restored':
                        x.copy_(saved_x);weight.copy_(saved_weight)
                        if bias is not None:bias.copy_(saved_bias)
                refresh();validate(mode)
            assert torch.equal(rng,torch.cuda.get_rng_state())
            assert torch.equal(x.view(torch.int32),saved_x.view(torch.int32)) and torch.equal(weight.view(torch.int32),saved_weight.view(torch.int32))
            numeric=all(d['candidate_FP64']['relative_l2']<6e-6 and (d['native_raw_FP64']['relative_l2']<6e-6 if odd else d['native_FP64']['relative_l2']<6e-6 and d['vs_native']['relative_l2']<1e-5) for d in dynamic)
            pairs=timings(baseline,graph) if not odd and numeric else []
            row=dict(module=name,input_shape=list(x.shape),weight_shape=list(weight.shape),odd=odd,config=tile,
                explicit_RNA_initial_input_scaled_Half_exact=input_rep,explicit_RNA_initial_weight_scaled_Half_exact=weight_rep,
                native_FP64=native_error,dynamic=dynamic,numerical_gate=numeric,pairs=pairs,
                all_positive=bool(pairs) and all(v['saved_ms']>0 for v in pairs),
                median_saved_ms=statistics.median(v['saved_ms'] for v in pairs) if pairs else None,
                NaN_overwrite_canaries_fresh=True,RNG_unchanged=True,native_weight_cache_used=cached is not None,
                native_weight_cache_proof=cached.proof if cached is not None else None,
                resources=dict(registers=kernel.n_regs,spills=kernel.n_spills,shared=kernel.metadata['shared']))
            rows.append(row);print('HEAD_SCALED_HALF_209_ROW '+json.dumps({k:v for k,v in row.items() if k not in ('pairs','dynamic','native_weight_cache_proof')}),flush=True)
            graph.reset()
        if cached is not None:cached.verify_canaries();cached.close()
        baseline.reset()
    report=dict(passed=True,records=rows,unsupported_native_caches=unsupported,compiler_failures=failures,
        source_metadata=config,eligible=[r for r in rows if r['all_positive'] and r['numerical_gate']],
        numerical_candidate=True,no_full_video_quality_claim=True,no_official_score=True,
        all_measured_baselines_have_original_native_weight_cache=True)
    assert len(rows)+len(failures)+len(unsupported)*len(config['tiles'])==len(config['modules'])*len(config['tiles'])+4
    Path('head-scaled-half-209-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HEAD_SCALED_HALF_209_CHECK passed=true',flush=True)

if __name__=='__main__':main()
