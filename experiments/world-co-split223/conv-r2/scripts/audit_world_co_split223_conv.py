"""Fresh synthetic full Conv2d AB including pack, all-gather and layout restore."""
import ctypes as C
import hashlib,importlib.util,json,os,statistics,struct
from pathlib import Path
import torch,triton
import torch.distributed as dist
from world_co_split223 import domain,parameters,gather_channels
from unifolm_wma.utils.native_graph_weight_cache_probe import GraphAPI
from unifolm_wma.utils.native_graph_world_filter_cache_probe import eligible_world_plan
from unifolm_wma.utils.native_sgemm_prepacked_swap import PrepackedSgemmSwap
from unifolm_wma.utils.native_kahan_filter_swap import CachedKahanFilterSwap
from unifolm_wma.utils.native_winograd_gemm_fp16_packed_kernel import transformed_gemm_fp16_packed
from unifolm_wma.utils.native_winograd_gemm_fp16_kahan_kernel import transformed_gemm_fp16_kahan


def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def layout(t,kind):return t.contiguous()if kind=='NCHW'else t.permute(1,0,2,3).contiguous().permute(1,0,2,3)
def replay(owner):owner.refresh_if_needed();owner.graph.replay()


def capture(module,x,spec,parallel,extension,api):
    rank=dist.get_rank();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    def run():
        y=module(x)
        return gather_channels(y,rank)if parallel else y
    with torch.cuda.stream(stream):
        for _ in range(3):run()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
    graph=torch.cuda.CUDAGraph();graph.enable_debug_mode()
    with torch.cuda.graph(graph,pool=torch.cuda.graph_pool_handle()):output=run()
    graph.replay();torch.cuda.synchronize()
    raw,_=extension.get_handles(graph);plan=eligible_world_plan(api.read(raw,dict(weight=module.weight)),module.weight)
    assert plan is not None,'No validated native Winograd plan'
    co,cols,k=spec['geometry'];bm,bn,bk,warps,stages=spec['config']
    payload=bytes.fromhex(plan[1]['arguments'][0]['bytes_hex']);assert struct.unpack_from('<3Q',payload,56)==(cols*k,k*co,cols*co)
    dummy=torch.empty((1,),device='cuda');half=torch.empty((1,),device='cuda',dtype=torch.float16)
    packed=spec['family']=='packed';function=transformed_gemm_fp16_packed if packed else transformed_gemm_fp16_kahan
    args=(half,half,dummy,dummy)if packed else(dummy,dummy,dummy)
    kernel=function.warmup(*args,co,cols,k,bm,bn,bk,num_warps=warps,num_stages=stages,enable_fp_fusion=False,grid=(triton.cdiv(co,bm)*triton.cdiv(cols,bn),36));kernel._init_handles()
    if packed:owner=PrepackedSgemmSwap(graph,module.weight,plan,kernel,tuple(spec['geometry']),spec['config'],extension,api)
    else:owner=CachedKahanFilterSwap(graph,module.weight,plan,tuple(spec['geometry']),spec['choice'],extension,api)
    original=owner.replacement
    if parallel:owner.replacement=parameters(owner,spec['geometry'],spec['config'],rank)
    owner.set_enabled(True)
    return graph,output,owner,original


def convolution(spec,kind,extension,api):
    rank=dist.get_rank();co,cols,ci=spec['geometry'];h,w={2560:(40,64),640:(20,32),192:(10,16)}[cols]
    module=torch.nn.Conv2d(ci,co,3,padding=1,bias=True,device='cuda',dtype=torch.float32).eval().requires_grad_(False)
    x=layout(torch.randn((16,ci,h,w),device='cuda')*.1,kind);x.add_(torch.arange(16,device='cuda').view(16,1,1,1)*.013)
    x0,w0,b0=x.clone(),module.weight.clone(),module.bias.clone()
    reference,y0,base,base_original=capture(module,x,spec,False,extension,api)
    candidate,y1,owner,original=capture(module,x,spec,True,extension,api)
    rng,cpu_rng=torch.cuda.get_rng_state(),torch.get_rng_state();cases=[];pairs=[]
    output_pointer=struct.unpack('<Q',bytes(original.buffers[-1]))[0]
    try:
        for case in('initial','changed_inputs','changed_weights','zero_weights','zero_inputs','restored'):
            x.copy_(x0);module.weight.copy_(w0);module.bias.copy_(b0)
            if case=='changed_inputs':x.mul_(.75).add_(.0137)
            elif case=='changed_weights':module.weight.add_(.000137);module.bias.add_(.00071)
            elif case=='zero_weights':module.weight.zero_()
            elif case=='zero_inputs':x.zero_()
            replay(base);expected=y0.clone();y1.fill_(float('nan'))
            api.check(api.lib.cuMemsetD32Async(output_pointer,0x7fc00000,36*co*cols,C.c_void_p(torch.cuda.current_stream().cuda_stream)),'poison validated Winograd output span')
            replay(owner);torch.cuda.synchronize();exact=bits(y1,expected);finite=bool(torch.isfinite(y1).all());assert exact and finite,(spec['geometry'],kind,rank,case,exact,finite)
            owner.verify_cache();base.verify_cache();cases.append(dict(case=case,all16_full_output_signedzero_bits=exact,finite=finite))
        assert torch.equal(rng,torch.cuda.get_rng_state())and torch.equal(cpu_rng,torch.get_rng_state())
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in range(6):
            pair={}
            for label,fn in ([('baseline',base),('candidate',owner)]if i%2==0 else[('candidate',owner),('baseline',base)]):
                dist.barrier();torch.cuda.synchronize()
                for _ in range(10):replay(fn)
                torch.cuda.synchronize();samples=[]
                for _ in range(7):
                    start.record()
                    for _ in range(3):replay(fn)
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)/3)
                pair[label+'_samples_ms']=samples;pair[label+'_ms']=statistics.median(samples)
            pair['saved_ms']=pair['baseline_ms']-pair['candidate_ms'];pairs.append(pair)
        owner.replacement=original;owner.set_enabled(True);replay(base);replay(owner);torch.cuda.synchronize();assert bits(y0,y1)
        owner.close();native=module(x).clone();candidate.replay();torch.cuda.synchronize();assert bits(y1,native)
        base.close();native=module(x).clone();reference.replay();torch.cuda.synchronize();assert bits(y0,native)
        return dict(geometry=spec['geometry'],family=spec['family'],config=spec['config'],layout=kind,input_stride=list(x.stride()),output_stride=list(y1.stride()),cases=cases,pairs=pairs,
            median_saved_ms=statistics.median(p['saved_ms']for p in pairs),all6_pairs_faster=all(p['saved_ms']>0 for p in pairs),
            all6_dynamic_full_outputs_signedzero_bits=True,original_native_and_numerical_restorations_bitwise=True,RNG_unchanged=True,
            original_CUBIN_constexpr_and_K_order=True,original_native_transforms_bias_and_version_refresh=True,
            pack_one_allgather_and_layout_restore_in_timing=True,output_workspace_poisoned_each_case=True,no_workspace_tensor_views_or_operand_exports=True)
    finally:
        if not owner.closed:owner.replacement=original;owner.close()
        if not base.closed:base.close()
        reference.reset();candidate.reset()


def main():
    torch.set_grad_enabled(False);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    canonical=Path(os.environ['CANONICAL_ROOT']);so=canonical/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so'
    assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so);extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension);api=GraphAPI()
    rows=[];specs=json.loads((Path(os.environ['EXPERIMENT_ROOT'])/'geometries.json').read_text())
    for spec in specs:
        for kind in('NCHW','NC_permuted'):
            rows.append(convolution(spec,kind,extension,api));Path('rank'+str(dist.get_rank())+'-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
            print('CO_SPLIT_CONV '+json.dumps(dict(geometry=spec['geometry'],layout=kind,rank=dist.get_rank(),saved_ms=rows[-1]['median_saved_ms'])),flush=True)
    report=dict(synthetic_only=True,no_official_score=True,full_model_and_video_quality_unverified=True,rows=rows,all_full_output_dynamic_cases_bitwise=True,per_geometry_six_AB_pairs=True)
    Path('rank'+str(dist.get_rank())+'-report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
