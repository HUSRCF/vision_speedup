"""Synthetic GPU matrix checks for half-frame packed/Kahan Winograd GEMMs.

No model checkpoint, real operands, native workspace wrappers or driver-node
mutation. Both reference and candidate use the original numerical arithmetic;
this is not a comparison against full-precision native convolution.
"""
import ctypes as C
import hashlib,importlib.util,json,os,statistics
from pathlib import Path
import torch
import triton
from guard_single_tmp209 import announce_worker
if os.environ.get('PREFIX_COLUMN_LAYOUT')=='frame_interleaved':
    from world_native_frame_interleaved209_kernel import (
        transformed_gemm_fp16_packed_frame_interleaved as transformed_gemm_fp16_packed_frame_prefix,
        transformed_gemm_fp16_kahan_frame_interleaved as transformed_gemm_fp16_kahan_frame_prefix)
else:
    from world_native_frame_prefix209_kernel import (
        transformed_gemm_fp16_packed_frame_prefix,transformed_gemm_fp16_kahan_frame_prefix)
from unifolm_wma.utils.native_winograd_gemm_fp16_packed_kernel import (
    transformed_gemm_fp16_packed,pack_native_filter)
from unifolm_wma.utils.native_winograd_gemm_fp16_kahan_kernel import transformed_gemm_fp16_kahan
from unifolm_wma.utils.native_graph_weight_cache_probe import GraphAPI


def bits(a,b):
    return torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


def witness(graph,kernel,pointers,grid,warps,extension,api):
    raw,_=extension.get_handles(graph)
    nodes=api.read(raw,{})['nodes']
    assert len(nodes)==1 and nodes[0]['type']==0
    node=nodes[0]
    assert int(node['function'],16)==kernel.function
    assert node['grid']==[*grid,1] and node['block']==[32*warps,1,1]
    assert node['shared_bytes']==kernel.shared
    assert [arg['size'] for arg in node['arguments']]==[8]*len(pointers)
    assert [int.from_bytes(bytes.fromhex(arg['bytes_hex']),'little') for arg in node['arguments']]==pointers
    for index in range(len(pointers)+1):
        offset,size=C.c_size_t(),C.c_size_t()
        status=api.lib.cuFuncGetParamInfo(C.c_void_p(kernel.function),index,C.byref(offset),C.byref(size))
        if index==len(pointers):assert status==1,'Unexpected extra runtime argument'
        else:
            api.check(status,'fresh prefix kernel parameter metadata')
            assert (offset.value,size.value)==(index*8,8)
    assert kernel.num_ctas==1 and tuple(kernel.cluster_dims)==(1,1,1)
    return dict(argument_sizes=[8]*len(pointers),bounded_pointer_ABI_verified=True,
        all_captured_argument_bytes_match_known_Torch_pointers=True,
        function_grid_block_shared_verified=True,new_kernel_cubin_SHA256=hashlib.sha256(kernel.asm['cubin']).hexdigest(),
        no_argument_bytes_or_addresses_exported=True)


def fixture(spec,extension,api):
    co,cols,k=spec['geometry'];bm,bn,bk,warps,stages=spec['config']
    active=cols//2;rounded=min(cols,triton.cdiv(active,bn)*bn)
    interleaved=os.environ.get('PREFIX_COLUMN_LAYOUT')=='frame_interleaved'
    selected=torch.arange(cols,device='cuda').view(-1,16)[:,:8].reshape(-1) if interleaved else None
    unselected=torch.ones(cols,device='cuda',dtype=torch.bool) if interleaved else None
    if interleaved:unselected[selected]=False
    shape=(36,co,cols)
    w=torch.randn((36,co,k),device='cuda')*.1
    x=torch.randn((36,k,cols),device='cuda')*.1
    saved_w,saved_x=w.clone(),x.clone()
    reference=torch.empty(shape,device='cuda')
    guard=4096
    backing=torch.full((reference.numel()+2*guard,),90909.,device='cuda')
    output=backing[guard:-guard].view(shape)
    if spec['family']=='packed':
        wh=torch.empty_like(w,dtype=torch.float16);wl=torch.empty_like(wh)
        def pack():
            pack_native_filter[(triton.cdiv(w.numel(),1024),)](w,wh,wl,w.numel(),1024,
                num_warps=4,enable_fp_fusion=False)
        pack()
        original,candidate=transformed_gemm_fp16_packed,transformed_gemm_fp16_packed_frame_prefix
        args=[wh,wl,x]
    else:
        original,candidate=transformed_gemm_fp16_kahan,transformed_gemm_fp16_kahan_frame_prefix
        args=[w,x]
        def pack():pass
    full_grid=(triton.cdiv(co,bm)*triton.cdiv(cols,bn),36)
    prefix_grid=(triton.cdiv(co,bm)*triton.cdiv(active,bn),36)
    config=dict(num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    def launch_original():return original[full_grid](*args,reference,co,cols,k,bm,bn,bk,**config)
    def launch_candidate():return candidate[prefix_grid](*args,output,co,cols,k,bm,bn,bk,active,**config)
    compiled_original=launch_original();compiled=launch_candidate();torch.cuda.synchronize()
    graph0,graph1=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
    graph0.enable_debug_mode();graph1.enable_debug_mode()
    with torch.cuda.graph(graph0):launch_original()
    with torch.cuda.graph(graph1):launch_candidate()
    compiled._init_handles()
    proof=witness(graph1,compiled,[p.data_ptr() for p in [*args,output]],prefix_grid,warps,extension,api)
    proof['reference_fresh_cubin_SHA256']=hashlib.sha256(compiled_original.asm['cubin']).hexdigest()
    cpu_rng,gpu_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
    cases=[]
    try:
        for name in ('initial','changed_inputs','changed_weights','zero_weights','zero_inputs','restored'):
            w.copy_(saved_w);x.copy_(saved_x)
            if name=='changed_inputs':x.mul_(.75).add_(.0137)
            elif name=='changed_weights':w.add_(.000137)
            elif name=='zero_weights':w.zero_()
            elif name=='zero_inputs':x.zero_()
            pack();output.fill_(float('nan'))
            graph0.replay();graph1.replay();torch.cuda.synchronize()
            a,b=(output[:,:,selected],reference[:,:,selected]) if interleaved else (output[:,:,:active],reference[:,:,:active])
            exact=bits(a,b);finite=bool(torch.isfinite(a).all())
            intact=bool(torch.isnan(output[:,:,unselected]).all()) if interleaved else (bool(torch.isnan(output[:,:,rounded:]).all()) if rounded<cols else True)
            canaries=bool(torch.all(backing[:guard]==90909.) and torch.all(backing[-guard:]==90909.))
            record=dict(case=name,prefix_signedzero_bits=exact,finite=finite,
                untouched_tail_NaN=intact,output_canaries=canaries,
                max_abs=float((a-b).abs().max()) if finite else None)
            cases.append(record)
            assert exact and finite and intact and canaries,(spec,record)
        pairs=[]
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for index in range(6):
            pair={}
            for label in (('original','prefix') if index%2==0 else ('prefix','original')):
                graph=graph0 if label=='original' else graph1
                for _ in range(10):graph.replay()
                torch.cuda.synchronize();samples=[]
                for _ in range(7):
                    start.record()
                    for _ in range(3):graph.replay()
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)/3)
                pair[label+'_samples_ms']=samples;pair[label+'_ms']=statistics.median(samples)
            pair['saved_ms']=pair['original_ms']-pair['prefix_ms'];pairs.append(pair)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        return dict(**spec,cases=cases,pairs=pairs,new_kernel_witness=proof,
            tested_column_mapping='interleaved_frames_first8_of16' if interleaved else 'contiguous_half_columns',
            all6_prefix_signedzero_bit_exact=True,full_COLS_strides_and_masks_preserved=True,
            active_cols=active,rounded_written_prefix_cols=rounded,
            original_grid=list(full_grid),prefix_grid=list(prefix_grid),RNG_unchanged=True,
            all6_pairs_faster=all(p['saved_ms']>0 for p in pairs),
            median_saved_ms=statistics.median(p['saved_ms'] for p in pairs),
            timing_excludes_common_static_weight_pack=True,
            not_a_complete_convolution_or_full_workload_speedup=True)
    finally:
        graph0.reset();graph1.reset()


def main():
    hold=announce_worker()
    torch.set_grad_enabled(False)
    assert torch.__version__=='2.3.1+cu121' and triton.__version__=='2.3.1'
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    canonical=Path(os.environ['CANONICAL_ROOT'])
    so=canonical/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so'
    assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    module_spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so)
    extension=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(extension)
    api=GraphAPI();rows=[]
    specs=json.loads((Path(os.environ['EXPERIMENT_ROOT'])/'matrix-geometries.json').read_text())
    for spec in specs:
        row=fixture(spec,extension,api);rows.append(row)
        Path('progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('WORLD_NATIVE_PREFIX_MATRIX '+json.dumps(row),flush=True)
    assert len(rows)==14
    report=dict(diagnostic_complete=True,synthetic_only=True,rows=rows,
        all_prefix_outputs_signedzero_bits=True,dynamic_checks=14*6,
        fresh_graph_and_bounded_CUBIN_pointer_ABI_witnesses=14,
        no_checkpoint_real_operands_native_workspace_wrappers_or_node_mutations=True,
        native_transform_frame_order_full_model_and_video_quality_unverified=True,
        original_production_cache_not_redirected_or_modified=True,
        isolated_new_experiment_cache=os.environ['TRITON_CACHE_DIR'],
        isolated_new_executable_launcher_cache=os.environ['TRITON_EXECUTABLE_CACHE_DIR'],
        no_official_score=True)
    Path('comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('WORLD_NATIVE_PREFIX_MATRIX_CHECK passed=true',flush=True)


if __name__=='__main__':main()
