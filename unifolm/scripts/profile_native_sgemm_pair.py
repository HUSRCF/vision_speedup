"""Separate numerical real-model probe, never an exact-output score."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import statistics
import struct
import sys
import time
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from torch.utils._pytree import tree_flatten
import triton
from native_graph_weight_cache_probe import GraphAPI
from native_graph_world_filter_cache_probe import FILTER,eligible_world_plan
from native_winograd_gemm_fp16_kernel import transformed_gemm_fp16
from native_sgemm_direct_params import DirectSgemmSwap
from native_sgemm_prepacked_swap import PrepackedSgemmSwap,NativeOnlyFilterSwap
from native_winograd_gemm_fp16_packed_kernel import transformed_gemm_fp16_packed
from torch.utils._pytree import tree_map
from benchmark_native_winograd_sgemm_audit import metrics
from unifolm_wma.utils.cuda_graph import DenoiserGraph
import unifolm_wma.utils.inference_runtime as runtime


class Complete(Exception):pass


def compare(left,right):
    aa,sa=tree_flatten(left);bb,sb=tree_flatten(right);assert sa==sb
    rows=[]
    for a,b in zip(aa,bb):
        if not isinstance(a,torch.Tensor):assert a==b;continue
        assert a.shape==b.shape and a.dtype==b.dtype
        e=metrics(a,b.double())
        e['first_input_diagnostic_bound_passed']=e['finite'] and e['relative_l2']<=.001 and e['max_abs']<=.001*max(1.,float(b.abs().max()))
        rows.append(e)
    assert len(rows)==3 and all(e['first_input_diagnostic_bound_passed'] for e in rows),rows
    return rows


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--diagnostic-dir',type=Path,required=True)
    p.add_argument('--profile-stage',choices=['world'],required=True)
    args,inference=p.parse_known_args();args.diagnostic_dir.mkdir(exist_ok=False,parents=True)
    assert torch.__version__=='2.3.1+cu121'
    config=json.loads(Path('native-sgemm-configs.json').read_text())
    configs={tuple(r['geometry']):r['config'] for r in config}
    provenance=json.loads(Path('graph-handles-provenance.json').read_text());so=Path(provenance['so'])
    assert hashlib.sha256(so.read_bytes()).hexdigest()==provenance['so_sha256']
    spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so)
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    api=GraphAPI();old_configure=runtime.configure_denoiser

    def configure(model,*aa,**kk):
        old_configure(model,*aa,**kk);original=model.apply_model
        assert isinstance(original,DenoiserGraph)
        def diagnose(*values,**keywords):
            if torch.backends.cudnn.allow_tf32:return original(*values,**keywords)
            assert not torch.backends.cuda.matmul.allow_tf32
            swaps=[];references=[];templates={};graph=None
            init=torch.cuda.CUDAGraph.__init__
            def retained(obj,*a,**k):init(obj,*a,**k);obj.enable_debug_mode()
            try:
                graph=DenoiserGraph(original.original,condition_refresher=original.condition_refresher,skip_unchanged_inputs=False)
                with patch.object(torch.cuda.CUDAGraph,'__init__',retained):baseline=graph(*values,**keywords)
                assert len(graph.entries)==1
                entry=next(iter(graph.entries.values()));native=entry['graph']
                assert entry.get('native_weight_cache') is None,'World cache and SGEMM must not silently share an executable pointer update'
                raw,_=extension.get_handles(native);inventory=api.read(raw,{})
                reference_kernels={}
                nodes={n['node']:n for n in inventory['nodes']};edges=inventory['edges']
                masters={w.data_ptr():(name,w) for name,w in model.named_parameters() if w.dtype==torch.float32 and w.is_cuda and w.is_contiguous() and w.ndim==4}
                chosen={};plans=[];started=time.perf_counter()
                for n in nodes.values():
                    if n.get('name')!=FILTER or [a['size'] for a in n.get('arguments',[])]!=[40]:continue
                    ptr=struct.unpack_from('<Q',bytes.fromhex(n['arguments'][0]['bytes_hex']),16)[0]
                    if ptr not in masters:continue
                    name,w=masters[ptr]
                    outgoing=[b for a,b in edges if a==n['node']]
                    view=dict(nodes=[dict(n,matched_tensors=['weight']),*[nodes[b] for b in outgoing]],edges=[[n['node'],b] for b in outgoing])
                    plan=eligible_world_plan(view,w)
                    if plan is None:continue
                    _,consumer,_,_=plan;payload=bytes.fromhex(consumer['arguments'][0]['bytes_hex'])
                    co,ci=w.shape[:2];stride=struct.unpack_from('<Q',payload,56)[0]
                    assert stride%ci==0;m=stride//ci;geometry=(co,m,ci)
                    if geometry not in configs:
                        swap=NativeOnlyFilterSwap(native,w,plan,extension,api)
                        swaps.append(swap);chosen[ptr]=w
                        plans.append(dict(weight=name,geometry=geometry,config=None,**swap.proof))
                        continue
                    a,b,c=struct.unpack_from('<3Q',payload)
                    assert b-a==36*m*ci*4 and c-b==36*co*ci*4
                    assert struct.unpack_from('<3Q',payload,56)==(m*ci,ci*co,m*co)
                    if geometry not in templates:
                        bm,bn,bk,warps,stages=configs[geometry]
                        dummy=torch.empty((1,),device=w.device)
                        half_dummy=torch.empty((1,),device=w.device,dtype=torch.float16)
                        kernel=transformed_gemm_fp16_packed.warmup(half_dummy,half_dummy,dummy,dummy,co,m,ci,bm,bn,bk,
                            num_warps=warps,num_stages=stages,enable_fp_fusion=False,
                            grid=(triton.cdiv(co,bm)*triton.cdiv(m,bn),36))
                        kernel._init_handles()
                        templates[geometry]=(None,half_dummy,dummy,kernel)
                        reference_kernel=transformed_gemm_fp16.warmup(dummy,dummy,dummy,co,m,ci,bm,bn,bk,
                            num_warps=warps,num_stages=stages,enable_fp_fusion=False,
                            grid=(triton.cdiv(co,bm)*triton.cdiv(m,bn),36))
                        reference_kernel._init_handles();reference_kernels[geometry]=reference_kernel
                    kernel=templates[geometry][-1]
                    references.append(DirectSgemmSwap(native,consumer,reference_kernels[geometry],geometry,configs[geometry],extension,api))
                    swap=PrepackedSgemmSwap(native,w,plan,kernel,geometry,configs[geometry],extension,api)
                    swaps.append(swap);chosen[ptr]=w
                    plans.append(dict(weight=name,geometry=geometry,config=configs[geometry],**swap.proof))
                assert swaps,'No proven microbenchmark geometry selected'
                torch.cuda.synchronize();setup_ms=(time.perf_counter()-started)*1000
                def enabled(on):
                    for reference in references:reference.set_enabled(False)
                    for swap in swaps:swap.set_enabled(on)
                def optimized(on):
                    enabled(False)
                    if on:
                        for swap in swaps:swap.set_enabled(True)
                    else:
                        for reference in references:reference.set_enabled(True)
                def previous_bits():
                    optimized(False);native.replay()
                    expected=tree_map(lambda v:v.clone() if isinstance(v,torch.Tensor) else v,entry['outputs'])
                    optimized(True);native.replay()
                    aa,sa=tree_flatten(entry['outputs']);bb,sb=tree_flatten(expected);assert sa==sb
                    for actual,reference in zip(aa,bb):
                        if isinstance(actual,torch.Tensor):
                            assert actual.dtype==reference.dtype==torch.float32
                            assert torch.equal(actual.contiguous().view(torch.int32),reference.contiguous().view(torch.int32))
                        else:assert actual==reference
                    for swap in swaps:swap.verify_cache()
                    return True
                enabled(True);native.replay();initial=compare(entry['outputs'],baseline);previous_bits()
                changed=tuple(v+.0001 if isinstance(v,torch.Tensor) and v.is_floating_point() else v for v in values)
                enabled(False);expected=graph(*changed,**keywords)
                enabled(True);updated=compare(graph(*changed,**keywords),expected);previous_bits()
                saved={ptr:w.clone() for ptr,w in chosen.items()}
                for w in chosen.values():w.add_(.00013)
                enabled(False);expected=graph(*changed,**keywords)
                enabled(True);mutated=compare(graph(*changed,**keywords),expected);previous_bits()
                for ptr,w in chosen.items():w.copy_(saved[ptr])
                enabled(False);restored=graph(*values,**keywords)
                aa,_=tree_flatten(restored);bb,_=tree_flatten(baseline)
                for a,b in zip(aa,bb):assert torch.equal(a.view(torch.int32),b.view(torch.int32))
                enabled(True);compare(graph(*values,**keywords),baseline);previous_bits()
                pairs=[];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for i in range(6):
                    row={}
                    for on in ([False,True] if i%2==0 else [True,False]):
                        optimized(on);samples=[]
                        for _ in range(3):
                            start.record()
                            for _ in range(10):
                                if on:
                                    for swap in swaps:swap.refresh_if_needed()
                                native.replay()
                            end.record();end.synchronize();samples.append(start.elapsed_time(end)/10)
                        row['candidate_ms' if on else 'baseline_ms']=statistics.median(samples)
                    row['saved_ms']=row['baseline_ms']-row['candidate_ms'];pairs.append(row)
                enabled(False);native.replay();compare(entry['outputs'],baseline)
                assert inventory==api.read(raw,{})
                summary=dict(stage='world',numerical_candidate=True,selected=len(swaps),plans=plans,pairs=pairs,
                    median_saved_ms=statistics.median(r['saved_ms'] for r in pairs),initial_outputs=initial,changed_input_outputs=updated,mutated_master_outputs=mutated,
                    first_input_bound_only=True,full_video_PSNR_unverified=True,original_native_transforms=True,
                    original_graph_definition_unchanged=True,restored_native_output_bits=True,source_inference_unchanged=True,
                    setup_ms=setup_ms,prepacked_weights=True,reference_current_FP16_bits=True,pair_baseline='current_FP16x3',
                    numerical_selected=len(references),native_only_selected=len(swaps)-len(references),
                    cached_native_filter_bytes=sum(s.cache.cache.numel()*4 for s in swaps),
                    packed_filter_bytes=sum(s.wh.numel()*4 for s in swaps if isinstance(s,PrepackedSgemmSwap)),
                    direct_params=True,template_bytes=sum(row[1].numel()*2+row[2].numel()*4 for row in templates.values()),
                    same_process=True,same_gpu=True,same_model_weights=True,same_actual_first_denoiser_input=True,no_official_score=True)
                (args.diagnostic_dir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
                print('NATIVE_SGEMM_PAIR '+json.dumps(summary),flush=True)
            finally:
                torch.cuda.synchronize()
                for reference in references:reference.close()
                for swap in swaps:swap.close()
                for cg,*_ in templates.values():
                    if cg is not None:cg.reset()
                for wrapper in [original,graph]:
                    if wrapper is None:continue
                    for e in wrapper.entries.values():
                        if e.get('native_weight_cache') is not None:e['native_weight_cache'].close()
                        e['graph'].reset()
                    wrapper.entries.clear()
            raise Complete
        model.apply_model=diagnose
    runtime.configure_denoiser=configure
    path=Path(__file__).parent/'evaluation/world_model_interaction.py'
    sys.path.insert(0,str(path.parent.resolve()));sys.argv=[str(path),*inference]
    try:runpy.run_path(str(path),run_name='__main__')
    except Complete:pass
    finally:runtime.configure_denoiser=old_configure


if __name__=='__main__':main()
