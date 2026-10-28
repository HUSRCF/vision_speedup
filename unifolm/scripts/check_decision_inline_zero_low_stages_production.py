"""Four actual inline geometries, two BM choices, one odd masked fixture."""
import json
from pathlib import Path
import statistics
import torch
import triton
from contextlib import contextmanager
import os
from unifolm_wma.utils import decision_inline_zero_low_stages_runtime as runtime
from unifolm_wma.utils import triton_decision_inline_zero_low


def install_fixture(configs):
    assert configs==runtime.CONFIGS and not runtime.active.captures
    return runtime.active


def clear_fixture():
    runtime.active.captures.clear();runtime.active.fallbacks.clear()


@contextmanager
def guard_mode(on):
    previous=os.environ.get('UNIFOLM_DECISION_INLINE_ZERO_LOW_STAGES3')
    os.environ['UNIFOLM_DECISION_INLINE_ZERO_LOW_STAGES3']='1' if on else '0'
    try:yield
    finally:
        if previous is None:os.environ.pop('UNIFOLM_DECISION_INLINE_ZERO_LOW_STAGES3',None)
        else:os.environ['UNIFOLM_DECISION_INLINE_ZERO_LOW_STAGES3']=previous

from unifolm_wma.utils.triton_half_compensated_linear import _half_compensated_gemm,pack_half_weight
from unifolm_wma.utils.triton_decision_inline_zero_low import inline_zero_low_gemm


def exact(a,b):assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(1517);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    configs=json.loads(Path('decision-inline-zero-low-stages-selection.json').read_text());assert len(configs)==4
    controller=install_fixture(configs)
    configs.extend(dict(geometry=[19,23,13],config=[32,64,bk,32],baseline_config=[32,64,bk,32],has_bias=True,output_groups=1,num_stages=stage,odd=True) for bk in [32,64] for stage in [1,3])
    rows=[]
    for item in configs:
        m,n,k=item['geometry'];bm,bn,bk,group=item['config'];obm,obn,obk,ogroup=item['baseline_config']
        assert bm==obm and bn==obn==64 and bk==obk and group==ogroup==32 and item['num_stages'] in [1,3]
        master=(torch.randn((n,k),device='cuda')*.01).half().float()
        value=torch.randn((m,k),device='cuda')*.3;weight=pack_half_weight(master)
        count=torch.count_nonzero(weight[1]);ptr=count.data_ptr();assert int(count)==0
        bias=torch.randn(n,device='cuda')*.01 if item['has_bias'] else None
        dummy=weight[0] if bias is None else bias
        backings=[torch.full((m*n+128,),90909.,device='cuda') for _ in range(2)]
        y0,y1=[x[64:-64].view(m,n) for x in backings]
        grid=(triton.cdiv(m,bm)*triton.cdiv(n,bn),);oldgrid=(triton.cdiv(m,obm)*triton.cdiv(n,obn),)
        def old():
            with guard_mode(False):return inline_zero_low_gemm[oldgrid](value,*weight,dummy,count,y0,m,n,k,item['has_bias'],
                obm,obn,1.,256.,True,ogroup,obk,num_warps=4,num_stages=2,enable_fp_fusion=False)
        def new():
            with guard_mode(True):return inline_zero_low_gemm[grid](value,*weight,dummy,count,y1,m,n,k,item['has_bias'],
                bm,bn,1.,256.,True,group,bk,num_warps=4,num_stages=2,enable_fp_fusion=False)
        old_kernel=old();new_kernel=new();exact(y0,y1)
        saved_master=master.clone();saved_value=value.clone();rng=torch.cuda.get_rng_state()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):old();new()
        torch.cuda.current_stream().wait_stream(stream)
        graphs={}
        for label,call in [('baseline',old),('candidate',new)]:
            graph=torch.cuda.CUDAGraph();graphs[label]=graph
            with torch.cuda.graph(graph,stream=stream):call()
        dynamic=[]
        for mode in ['general','Half','zero','restored']:
            master.add_(.00013);value.add_(.001)
            if mode=='Half':master.copy_(master.half().float())
            if mode=='zero':master.zero_();value.zero_()
            if mode=='restored':master.copy_(saved_master);value.copy_(saved_value)
            pack_half_weight(master,out=weight);count.copy_(torch.count_nonzero(weight[1]));c=int(count)
            assert (c>0 if mode=='general' else c==0);dynamic.append(dict(mode=mode,nonzero=c))
            y0.fill_(float('nan'));y1.fill_(float('nan'))
            for graph in graphs.values():graph.replay()
            exact(y0,y1);assert torch.isfinite(y1).all() and count.data_ptr()==ptr
            fresh=torch.empty_like(y1)
            inline_zero_low_gemm[grid](value,*weight,dummy,count,fresh,m,n,k,item['has_bias'],
                bm,bn,1.,256.,True,group,bk,num_warps=4,num_stages=item['num_stages'],enable_fp_fusion=False)
            exact(y1,fresh)
            assert all(torch.all(x[:64]==90909.) and torch.all(x[-64:]==90909.) for x in backings)
        assert torch.equal(rng,torch.cuda.get_rng_state())
        for i in range(40):graphs['candidate' if i%2 else 'baseline'].replay()
        torch.cuda.synchronize();pairs=[]
        if not item.get('odd'):
            for i in range(6):
                pair={}
                for label in (['baseline','candidate'] if i%2==0 else ['candidate','baseline']):
                    graph=graphs[label];graph.replay();graph.replay();torch.cuda.synchronize();samples=[]
                    for _ in range(7):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10):graph.replay()
                        end.record();end.synchronize();samples.append(start.elapsed_time(end)/10)
                    pair[label+'_samples_ms']=samples;pair[label+'_ms']=statistics.median(samples)
                pair['saved_ms']=pair['baseline_ms']-pair['candidate_ms'];pairs.append(pair)
        row=dict(geometry=[m,n,k],config=item['config'],baseline_config=item['baseline_config'],has_bias=item['has_bias'],num_stages=item['num_stages'],fresh_destinations_exact=True,
            output_groups=1,nodes=item.get('nodes'),dynamic=dynamic,initial_general_Half_zero_restored_bits=True,
            NaN_overwrite=True,RNG_unchanged=True,Count_same_pointer=True,output_canaries_intact=True,
            all_input_splitting_inside_timed_GEMM=True,original_JITs_unchanged=True,Count_refresh_outside_replay=True,
            baseline_resources=dict(registers=old_kernel.n_regs,spills=old_kernel.n_spills,shared_bytes=old_kernel.metadata['shared']),
            candidate_resources=dict(registers=new_kernel.n_regs,spills=new_kernel.n_spills,shared_bytes=new_kernel.metadata['shared']),
            pairs=pairs,median_saved_ms=statistics.median(p['saved_ms'] for p in pairs) if pairs else None,no_model_or_full_quality_claim=True)
        rows.append(row);print('DECISION_INLINE_ZERO_LOW_STAGES_PRODUCTION '+json.dumps(row),flush=True)
        assert controller.report()['captured']==(0 if item.get('odd') else 1)
        if item.get('odd'):assert controller.fallbacks['unqualified']>0
        controller.captures.clear();controller.fallbacks.clear()
        for graph in graphs.values():graph.reset()
    clear_fixture();assert runtime.ORIGINAL_JIT.run is runtime.launched
    assert len(rows)==8
    print('DECISION_INLINE_ZERO_LOW_STAGES_PRODUCTION_CHECK passed=true fixtures=8 no_official_score=true',flush=True)


if __name__=='__main__':main()
