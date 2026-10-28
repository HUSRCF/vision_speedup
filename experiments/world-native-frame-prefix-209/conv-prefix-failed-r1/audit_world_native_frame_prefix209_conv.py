"""Fresh synthetic cuDNN/Winograd frame-order checks, without workspace views."""
import ctypes as C
import hashlib,importlib.util,json,os,statistics,struct
from pathlib import Path
import torch,triton
from guard_single_tmp209 import announce_worker
from audit_world_native_frame_prefix209_matrix import fixture,bits
from world_native_frame_prefix209_kernel import (
    transformed_gemm_fp16_packed_frame_prefix,transformed_gemm_fp16_kahan_frame_prefix)
from unifolm_wma.utils.native_winograd_gemm_fp16_packed_kernel import transformed_gemm_fp16_packed
from unifolm_wma.utils.native_winograd_gemm_fp16_kahan_kernel import transformed_gemm_fp16_kahan
from unifolm_wma.utils.native_graph_weight_cache_probe import GraphAPI,OwnedParams
from unifolm_wma.utils.native_graph_world_filter_cache_probe import eligible_world_plan
from unifolm_wma.utils.native_sgemm_prepacked_swap import PrepackedSgemmSwap
from unifolm_wma.utils.native_kahan_filter_swap import CachedKahanFilterSwap


def layout(t,kind):
    if kind=='NCHW':return t.contiguous()
    assert kind=='NC_permuted'
    return t.permute(1,0,2,3).contiguous().permute(1,0,2,3)


def captured(module,inputs,spec,matrix,extension,api):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):module(inputs)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph();graph.enable_debug_mode()
    with torch.cuda.graph(graph,pool=torch.cuda.graph_pool_handle()):output=module(inputs)
    graph.replay();torch.cuda.synchronize()
    raw,_=extension.get_handles(graph);inventory=api.read(raw,dict(weight=module.weight))
    plan=eligible_world_plan(inventory,module.weight);assert plan is not None,('No validated native plan',spec,list(inputs.stride()))
    _,node,_,_=plan;payload=bytes.fromhex(node['arguments'][0]['bytes_hex'])
    co,cols,k=spec['geometry'];bm,bn,bk,warps,stages=spec['config']
    assert struct.unpack_from('<3Q',payload,56)==(cols*k,k*co,cols*co)
    dummy=torch.empty((1,),device='cuda');half=torch.empty((1,),device='cuda',dtype=torch.float16)
    original=transformed_gemm_fp16_packed if spec['family']=='packed' else transformed_gemm_fp16_kahan
    args=(half,half,dummy,dummy) if spec['family']=='packed' else (dummy,dummy,dummy)
    kernel=original.warmup(*args,co,cols,k,bm,bn,bk,grid=(triton.cdiv(co,bm)*triton.cdiv(cols,bn),36),
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    kernel._init_handles()
    assert hashlib.sha256(kernel.asm['cubin']).hexdigest()==matrix['new_kernel_witness']['reference_fresh_cubin_SHA256']
    if spec['family']=='packed':
        owner=PrepackedSgemmSwap(graph,module.weight,plan,kernel,tuple(spec['geometry']),spec['config'],extension,api)
    else:
        owner=CachedKahanFilterSwap(graph,module.weight,plan,tuple(spec['geometry']),
            dict(config=spec['config'],witness_cubin_sha256=hashlib.sha256(kernel.kernel).hexdigest(),
                primitive_strict_conv_3e6_passed=False),extension,api)
    owner.set_enabled(True)
    return graph,output,owner,struct.unpack_from('<Q',payload,16)[0]


def prefix_parameters(owner,spec,matrix):
    co,cols,k=spec['geometry'];bm,bn,bk,warps,stages=spec['config']
    dummy=torch.empty((1,),device='cuda');half=torch.empty((1,),device='cuda',dtype=torch.float16)
    function=transformed_gemm_fp16_packed_frame_prefix if spec['family']=='packed' else transformed_gemm_fp16_kahan_frame_prefix
    args=(half,half,dummy,dummy) if spec['family']=='packed' else (dummy,dummy,dummy)
    compiled=function.warmup(*args,co,cols,k,bm,bn,bk,cols//2,
        grid=(triton.cdiv(co,bm)*triton.cdiv(cols//2,bn),36),num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    compiled._init_handles()
    assert hashlib.sha256(compiled.asm['cubin']).hexdigest()==matrix['new_kernel_witness']['new_kernel_cubin_SHA256']
    buffers=owner.replacement.buffers
    assert len(buffers)==(4 if spec['family']=='packed' else 3) and all(len(b)==8 for b in buffers)
    # These pointers are copied only from the original owner after its exact
    # native plan/span checks. Never dereference them or construct tensor views.
    params=OwnedParams(dict(function=hex(compiled.function),
        grid=[triton.cdiv(co,bm)*triton.cdiv(cols//2,bn),36,1],block=[warps*32,1,1],shared_bytes=compiled.shared,
        arguments=[dict(index=i,offset=i*8,size=8,bytes_hex=bytes(b).hex()) for i,b in enumerate(buffers)]))
    assert [bytes(b) for b in params.buffers]==[bytes(b) for b in buffers]
    return params,compiled


def convolution(spec,matrix,kind,extension,api):
    co,cols,ci=spec['geometry'];h,w={2560:(40,64),640:(20,32),192:(10,16)}[cols]
    module=torch.nn.Conv2d(ci,co,3,padding=1,bias=True,device='cuda',dtype=torch.float32).eval().requires_grad_(False)
    source=layout(torch.randn((16,ci,h,w),device='cuda')*.1,kind)
    source.add_(torch.arange(16,device='cuda').view(16,1,1,1)*.013)
    padded=layout(torch.zeros_like(source),kind)
    saved_x,saved_w,saved_b=source.clone(),module.weight.clone(),module.bias.clone()
    reference,y0,base,_=captured(module,source,spec,matrix,extension,api)
    candidate,y1,owner,output_pointer=captured(module,padded,spec,matrix,extension,api)
    full=owner.replacement;prefix,kernel=prefix_parameters(owner,spec,matrix)
    cpu_rng,gpu_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
    cases=[];timings=[]
    def select(on):
        owner.replacement=prefix if on else full;owner.set_enabled(True)
    try:
        select(True)
        for rank in (0,1):
            for case in ('initial','changed_inputs','changed_weights','zero_weights','zero_inputs','restored'):
                source.copy_(saved_x);module.weight.copy_(saved_w);module.bias.copy_(saved_b)
                if case=='changed_inputs':source.mul_(.75).add_(.0137)
                elif case=='changed_weights':module.weight.add_(.000137);module.bias.add_(.00071)
                elif case=='zero_weights':module.weight.zero_()
                elif case=='zero_inputs':source.zero_()
                padded.zero_();padded[:8].copy_(source[rank*8:(rank+1)*8])
                base.replay();expected=y0[rank*8:(rank+1)*8].clone()
                y1.fill_(float('nan'))
                # Span follows the original validated native SGEMM metadata;
                # poisoning prevents previous full-grid values from passing.
                api.check(api.lib.cuMemsetD32Async(output_pointer,0x7fc00000,36*co*cols,
                    C.c_void_p(torch.cuda.current_stream().cuda_stream)),'poison bounded native GEMM output')
                owner.replay();torch.cuda.synchronize()
                actual=y1[:8];exact=bits(actual,expected);finite=bool(torch.isfinite(actual).all())
                row=dict(rank=rank,case=case,first8_signedzero_bits=exact,finite=finite,
                    max_abs=float((actual-expected).abs().max()) if finite else None)
                cases.append(row);assert exact and finite,(spec,kind,row)
                owner.verify_cache();base.verify_cache()
            pairs=[];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for index in range(6):
                pair={}
                for label,on in ([('full',False),('prefix',True)] if index%2==0 else [('prefix',True),('full',False)]):
                    select(on)
                    for _ in range(10):owner.replay()
                    torch.cuda.synchronize();samples=[]
                    for _ in range(7):
                        start.record()
                        for _ in range(3):owner.replay()
                        end.record();end.synchronize();samples.append(start.elapsed_time(end)/3)
                    pair[label+'_samples_ms']=samples;pair[label+'_ms']=statistics.median(samples)
                pair['saved_ms']=pair['full_ms']-pair['prefix_ms'];pairs.append(pair)
            timings.append(dict(rank=rank,pairs=pairs,all6_pairs_faster=all(p['saved_ms']>0 for p in pairs),
                median_saved_ms=statistics.median(p['saved_ms'] for p in pairs)))
            select(True)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        select(False);source.copy_(padded)
        base.replay();owner.replay();torch.cuda.synchronize()
        assert bits(y0,y1),'Restore original numerical full16 graph'
        owner.close();expected=module(padded).clone();candidate.replay();torch.cuda.synchronize()
        assert bits(y1,expected),'Restore original native full16 graph'
        base.close();expected=module(source).clone();reference.replay();torch.cuda.synchronize()
        assert bits(y0,expected),'Restore reference native full16 graph'
        return dict(**spec,layout=kind,height=h,width=w,strides=list(padded.stride()),cases=cases,timings=timings,
            all12_dynamic_first8_signedzero_bits=True,RNG_unchanged=True,
            original_numerical_and_native_full16_restorations_signedzero_bits=True,
            new_kernel_matches_fresh_synthetic_capture_CUBIN=True,
            full_native_transforms_bias_and_cache_refresh_in_timing=True,
            output_workspace_poisoned_before_every_dynamic_case=True,
            no_native_workspace_tensor_wrappers_or_raw_operand_exports=True,
            full_model_and_video_quality_unverified=True)
    finally:
        if not owner.closed:
            owner.replacement=full;owner.close()
        if not base.closed:base.close()
        reference.reset();candidate.reset()


def main():
    hold=announce_worker();torch.set_grad_enabled(False)
    assert torch.__version__=='2.3.1+cu121' and triton.__version__=='2.3.1'
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    canonical=Path(os.environ['CANONICAL_ROOT'])
    so=canonical/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so'
    assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    module_spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so)
    extension=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(extension)
    api=GraphAPI();rows=[];matrix_rows=[]
    specs=json.loads((Path(os.environ['EXPERIMENT_ROOT'])/'matrix-geometries.json').read_text())
    for spec in specs:
        matrix=fixture(spec,extension,api);matrix_rows.append(matrix)
        for kind in ('NCHW','NC_permuted'):
            row=convolution(spec,matrix,kind,extension,api);rows.append(row)
            Path('progress.json').write_text(json.dumps(dict(rows=rows,matrix_rows=matrix_rows),indent=2)+'\n')
            print('WORLD_NATIVE_PREFIX_CONV '+json.dumps(row),flush=True)
    assert len(rows)==28 and len(matrix_rows)==14
    report=dict(diagnostic_complete=True,synthetic_only=True,rows=rows,matrix_rows=matrix_rows,
        all_prefix_outputs_signedzero_bits=True,dynamic_checks=28*12,matrix_dynamic_checks=14*6,
        all_native_and_numerical_full16_restorations_signedzero_bits=True,
        no_checkpoint_or_raw_operands=True,full_model_and_video_quality_unverified=True,no_official_score=True)
    Path('comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('WORLD_NATIVE_PREFIX_CONV_CHECK passed=true',flush=True)


if __name__=='__main__':main()
