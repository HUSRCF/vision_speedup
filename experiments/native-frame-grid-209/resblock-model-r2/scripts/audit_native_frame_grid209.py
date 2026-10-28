"""Bounded synthetic test: shrink only native Conv2d's spatial launch grid.

Keep N16 descriptors, all kernel argument bytes and split-K partitions intact.
Changing inputs and weights between checks prevents stale native workspace from
passing as computed output. No native workspace is wrapped or dereferenced.
"""
from datetime import datetime,timezone
import ctypes as C
import hashlib,importlib.util,json,math,statistics
from pathlib import Path
import torch
from audit_native_padded_frames209 import bits,layout
from unifolm_wma.utils.native_graph_weight_cache_probe import GraphAPI,OwnedParams,eligible_plan,CUTLASS,XMMA,XMMA_NONINDEXED
import unifolm_wma.utils.native_graph_weight_cache as production


def reduced_grid(node,co,h,w):
    grid=list(node['grid']);name=node['name']
    if name==CUTLASS:
        full_tiles=math.ceil(16*h*w/128);half_tiles=math.ceil(8*h*w/128)
        assert grid[0]%full_tiles==0
        swizzle=grid[0]//full_tiles
        assert swizzle in (1,2,4,8) and grid[1]==math.ceil(math.ceil(co/128)/swizzle)
        grid[0]=half_tiles*swizzle;axis=0
    else:
        assert name.startswith((XMMA,XMMA_NONINDEXED)),name
        # XMMA's first tile dimension covers flattened batch/spatial rows;
        # its second dimension covers output channels. Earlier fixtures used
        # square tiles only. The remaining-shape inventory also witnessed the
        # nonindexed 256x128x32 family, with grid [ceil(CO/128),ceil(NHW/256),Z].
        tiles=tuple(map(int,name.split('tilesize')[1].split('_')[0].split('x')))
        assert tiles in ((64,64,64),(128,128,16),(256,128,32)),tiles
        spatial_tile,channel_tile,_=tiles
        assert node['grid'][0]==math.ceil(co/channel_tile) and node['grid'][1]==math.ceil(16*h*w/spatial_tile)
        grid[1]=math.ceil(8*h*w/spatial_tile);axis=1
    assert all(0<n<=o for n,o in zip(grid,node['grid'])) and grid[2]==node['grid'][2]
    return grid,axis


def run(extension,api,co,ci,h,w,rank,kind):
    module=torch.nn.Conv2d(ci,co,3,padding=1,bias=True,device='cuda',dtype=torch.float32).eval()
    module.requires_grad_(False)
    source=layout(torch.randn((16,ci,h,w),device='cuda')*.1,kind)
    source+=torch.arange(16,device='cuda',dtype=torch.float32).view(16,1,1,1)*.013
    padded=layout(torch.zeros_like(source),kind)
    padded[:8].copy_(source[rank*8:(rank+1)*8])
    saved_x=source.clone();saved_w=module.weight.clone();saved_b=module.bias.clone()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):module(padded)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    pool=torch.cuda.graph_pool_handle();graph=torch.cuda.CUDAGraph();graph.enable_debug_mode()
    with torch.cuda.graph(graph,pool=pool):output=module(padded)
    graph.replay();torch.cuda.synchronize()
    raw,executable=extension.get_handles(graph)
    inventory=api.read(raw,dict(weight=module.weight));plan=eligible_plan(inventory,module.weight)
    assert plan is not None,(co,ci,h,w,rank,kind)
    transform,node,pointer_byte,old_filter,passthrough=plan
    grid,axis=reduced_grid(node,co,h,w)
    original=OwnedParams(node);candidate=OwnedParams(node)
    candidate.params.grid_x,candidate.params.grid_y,candidate.params.grid_z=grid
    assert [bytes(b) for b in original.buffers]==[bytes(b) for b in candidate.buffers]
    assert (candidate.params.func,candidate.params.block_x,candidate.params.block_y,candidate.params.block_z,candidate.params.shared)==(
        original.params.func,original.params.block_x,original.params.block_y,original.params.block_z,original.params.shared)
    def select(on):
        params=candidate if on else original
        api.check(api.lib.cuGraphExecKernelNodeSetParams(executable,int(node['node'],16),C.byref(params.params)),
            'set only native spatial grid')
    cases=[];pairs=[]
    try:
        select(True)
        for case in ('initial','changed_inputs','changed_weights','zero_weights','zero_inputs','restored'):
            source.copy_(saved_x);module.weight.copy_(saved_w);module.bias.copy_(saved_b)
            if case=='changed_inputs':source.mul_(.75).add_(.0137)
            elif case=='changed_weights':module.weight.add_(.000137);module.bias.add_(.00071)
            elif case=='zero_weights':module.weight.zero_()
            elif case=='zero_inputs':source.zero_()
            # Eager reference uses the normal allocator; the captured native
            # graph owns a distinct private pool and is NOT replayed at full
            # grid between cases. Its unwritten workspace retains old values.
            expected=module(source)[rank*8:(rank+1)*8].clone()
            assert expected.data_ptr()!=output.data_ptr()
            padded.zero_();padded[:8].copy_(source[rank*8:(rank+1)*8])
            output.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            actual=output[:8];passed=bits(actual,expected)
            finite=bool(torch.isfinite(actual).all())
            cases.append(dict(case=case,first8_signedzero_bits=passed,finite=finite,
                max_abs=float((actual-expected).abs().max()) if finite else None))
            if not passed:break
        all_passed=len(cases)==6 and all(c['first8_signedzero_bits'] for c in cases)
        if all_passed:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for index in range(6):
                pair={}
                for label,on in ([('baseline',False),('candidate',True)] if index%2==0
                        else [('candidate',True),('baseline',False)]):
                    select(on)
                    for _ in range(10):graph.replay()
                    torch.cuda.synchronize();samples=[]
                    for _ in range(7):
                        start.record()
                        for _ in range(10):graph.replay()
                        end.record();end.synchronize();samples.append(start.elapsed_time(end)/10)
                    pair[label+'_samples_ms']=samples;pair[label+'_ms']=statistics.median(samples)
                pair['saved_ms']=pair['baseline_ms']-pair['candidate_ms'];pairs.append(pair)
    finally:
        select(False);padded.zero_();padded[:8].copy_(source[rank*8:(rank+1)*8])
        expected=module(padded).clone();graph.replay();torch.cuda.synchronize()
        assert bits(output,expected),'Original native grid restoration must match all16 outputs'
    row=dict(co=co,ci=ci,height=h,width=w,rank=rank,layout=kind,strides=list(padded.stride()),
        consumer_name=node['name'],original_grid=node['grid'],reduced_grid=grid,axis=axis,
        all_kernel_argument_bytes_unchanged=True,function_block_shared_unchanged=True,
        full_N16_descriptors_and_split_K_preserved=True,native_memsets_unchanged=len(passthrough),
        all6_dynamic_cases_first8_signedzero_bit_exact=all_passed,cases=cases,pairs=pairs,
        all_pairs_faster=bool(pairs) and all(p['saved_ms']>0 for p in pairs),
        median_saved_ms=statistics.median(p['saved_ms'] for p in pairs) if pairs else None,
        original_full_grid_all16_restoration_bitwise=True,
        no_native_workspace_tensor_wrappers=True,no_full_model_or_parallel_speedup_claim=True)
    graph.reset();return row


def main():
    torch.set_grad_enabled(False);assert torch.__version__=='2.3.1+cu121'
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    directory=Path.home()/'benchmark-runs/diagnostics'/('native-frame-grid-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    directory.mkdir(exist_ok=False);print('DIAGNOSTIC_DIR='+str(directory),flush=True)
    so=Path(production.__file__).parent/'native-graph-handles/team07_cuda_graph_handles.so'
    assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so)
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    api=GraphAPI();rows=[]
    for co,ci,h,w in ((320,320,40,64),(640,640,20,32),(1280,1280,10,16),
                     (1280,1280,5,8),(640,1920,20,32),(320,960,40,64)):
        for kind in ('NCHW','NC_permuted'):
            for rank in (0,1):
                row=run(extension,api,co,ci,h,w,rank,kind);rows.append(row)
                print('NATIVE_FRAME_GRID '+json.dumps(row),flush=True)
                (directory/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert len(rows)==24
    report=dict(diagnostic_complete=True,synthetic_only=True,rows=rows,
        all24_first8_dynamic_cases_signedzero_bit_exact=all(r['all6_dynamic_cases_first8_signedzero_bit_exact'] for r in rows),
        all24_original_grid_restorations_signedzero_bit_exact=True,
        all_kernel_argument_bytes_unchanged=True,no_raw_operands_or_addresses_exported=True,
        no_full_model_or_video_quality_result=True,no_official_score=True)
    (directory/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_FRAME_GRID_CHECK passed=true',flush=True)


if __name__=='__main__':main()
