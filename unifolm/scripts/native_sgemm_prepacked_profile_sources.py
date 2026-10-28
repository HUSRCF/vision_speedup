"""Real-model prepacked-filter probe against unchanged current FP16 math."""
import ast
from native_sgemm_direct_profile_sources import profile as direct_profile


def profile():
    t=direct_profile()
    def one(old,new):
        nonlocal t
        assert t.count(old)==1,old
        t=t.replace(old,new)
    one('from native_sgemm_direct_params import DirectSgemmSwap',
        'from native_sgemm_direct_params import DirectSgemmSwap\n'
        'from native_sgemm_prepacked_swap import PrepackedSgemmSwap,NativeOnlyFilterSwap\n'
        'from native_winograd_gemm_fp16_packed_kernel import transformed_gemm_fp16_packed\n'
        'from torch.utils._pytree import tree_map')
    one('            swaps=[];templates={};graph=None',
        '            swaps=[];references=[];templates={};graph=None')
    one("                nodes={n['node']:n for n in inventory['nodes']};edges=inventory['edges']",
        "                reference_kernels={}\n                nodes={n['node']:n for n in inventory['nodes']};edges=inventory['edges']")
    one('                    if geometry not in configs:continue',
        '''                    if geometry not in configs:
                        swap=NativeOnlyFilterSwap(native,w,plan,extension,api)
                        swaps.append(swap);chosen[ptr]=w
                        plans.append(dict(weight=name,geometry=geometry,config=None,**swap.proof))
                        continue''')
    start=t.index('                        dummy=torch.empty(')
    end=t.index('                    swaps.append(swap)',start)
    t=t[:start]+'''                        dummy=torch.empty((1,),device=w.device)
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
'''+t[end:]
    one('                def enabled(on):\n                    for swap in swaps:swap.set_enabled(on)',
        '''                def enabled(on):
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
                    return True''')
    one("                enabled(True);native.replay();initial=compare(entry['outputs'],baseline)",
        "                enabled(True);native.replay();initial=compare(entry['outputs'],baseline);previous_bits()")
    one('                enabled(True);updated=compare(graph(*changed,**keywords),expected)',
        '                enabled(True);updated=compare(graph(*changed,**keywords),expected);previous_bits()')
    one('                enabled(True);mutated=compare(graph(*changed,**keywords),expected)',
        '                enabled(True);mutated=compare(graph(*changed,**keywords),expected);previous_bits()')
    one('                enabled(True);compare(graph(*values,**keywords),baseline)',
        '                enabled(True);compare(graph(*values,**keywords),baseline);previous_bits()')
    one('                        enabled(on);samples=[]','                        optimized(on);samples=[]')
    # Full-model cache refresh belongs to each candidate replay, as it will in
    # production. Calling the raw graph directly would hide Python overhead.
    one('                            for _ in range(10):native.replay()',
        '                            for _ in range(10):\n                                if on:\n                                    for swap in swaps:swap.refresh_if_needed()\n                                native.replay()')
    one('direct_params=True,template_bytes=sum(row[1].numel()*4 for row in templates.values()),',
        '''prepacked_weights=True,reference_current_FP16_bits=True,pair_baseline='current_FP16x3',
                    numerical_selected=len(references),native_only_selected=len(swaps)-len(references),
                    cached_native_filter_bytes=sum(s.cache.cache.numel()*4 for s in swaps),
                    packed_filter_bytes=sum(s.wh.numel()*4 for s in swaps if isinstance(s,PrepackedSgemmSwap)),
                    direct_params=True,template_bytes=sum(row[1].numel()*2+row[2].numel()*4 for row in templates.values()),''')
    one('                for swap in swaps:swap.close()',
        '                for reference in references:reference.close()\n                for swap in swaps:swap.close()')
    ast.parse(t);return t
