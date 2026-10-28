"""Exercise Count lifecycle through real pack/dispatch wrappers and fallback."""
import json
import torch
import triton
import unifolm_wma.utils.triton_half_compensated_linear as half
import unifolm_wma.utils.triton_prepacked_input_linear as prepacked
from contextlib import contextmanager
import os
from unifolm_wma.utils import world_zero_low_runtime as runtime


def install_fixture(configs):
    assert not runtime.active.owners
    assert all(any(all(c[k]==qualified[k] for k in ['geometry','config','has_bias','output_groups','num_stages'])
                   for qualified in runtime.CONFIGS) for c in configs)
    return runtime.active


def clear_fixture():
    runtime.active.owners.clear();runtime.active.captures.clear();runtime.active.fallbacks.clear()


@contextmanager
def guard_mode(on):
    previous=os.environ.get('UNIFOLM_WORLD_PREPACKED_ZERO_LOW')
    os.environ['UNIFOLM_WORLD_PREPACKED_ZERO_LOW']='1' if on else '0'
    try:yield
    finally:
        if previous is None:os.environ.pop('UNIFOLM_WORLD_PREPACKED_ZERO_LOW',None)
        else:os.environ['UNIFOLM_WORLD_PREPACKED_ZERO_LOW']=previous



@torch.no_grad()
def main():
    torch.manual_seed(1513);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    for fixture,has_bias in enumerate([True,False]):
        m,n,k=640,1280,1280;bm=32
        config=dict(geometry=[m,n,k],config=[bm,64,64,32],has_bias=has_bias,output_groups=1,num_stages=2)
        master=(torch.randn(n,k,device='cuda')*.01).half().float()
        value=torch.randn(m,k,device='cuda')*.2
        bias=torch.randn(n,device='cuda')*.01 if has_bias else None
        # Parts pre-exist before installation, as they do in the model pair.
        weight=half.pack_half_weight(master);inputs=half.pack_half_weight(value,weight_scale=1.)
        y0=torch.empty((m,n),device='cuda');y1=torch.empty_like(y0);grid=(triton.cdiv(m,bm)*triton.cdiv(n,64),)
        dummy=weight[0] if bias is None else bias
        controller=install_fixture([config])
        def call(out):return prepacked._prepacked_gemm[grid](*inputs,*weight,dummy,out,
            m,n,k,has_bias,bm,64,64,32,1,num_warps=4,num_stages=2,enable_fp_fusion=False)
        def old():
            with guard_mode(False):return call(y0)
        def new():
            with guard_mode(True):return call(y1)
        old();new();runtime.exact(y0,y1);assert len(controller.owners)==1
        owner=controller.owners[id(weight[1])];ptr=owner.pointer
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):old();new()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):new()
        rng=torch.cuda.get_rng_state();history=[]
        for mode in ['general','Half','zero','restored']:
            master.add_(.00013);value.add_(.001)
            if mode=='Half':master.copy_(master.half().float())
            if mode=='zero':master.zero_();value.zero_()
            if mode=='restored':master.copy_((master+.007).half().float())
            # Baseline-mode refresh must update already registered predicates.
            with guard_mode(False):half.pack_half_weight(master,out=weight)
            half.pack_half_weight(value,weight_scale=1.,out=inputs)
            count=int(owner.count);history.append(dict(mode=mode,nonzero=count))
            assert (count>0 if mode=='general' else count==0)
            y1.fill_(float('nan'));old();graph.replay();runtime.exact(y0,y1)
            assert torch.isfinite(y1).all() and owner.pointer==owner.count.data_ptr()==ptr
        # An unregistered weight pair encountered first during capture takes
        # the unchanged original kernel, without allocating a device predicate.
        fresh=tuple(x.clone() for x in weight)
        cold=torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(cold,stream=stream):
            with guard_mode(True):prepacked._prepacked_gemm[grid](*inputs,*fresh,dummy,y1,
                m,n,k,has_bias,bm,64,64,32,1,num_warps=4,num_stages=2,enable_fp_fusion=False)
        torch.cuda.current_stream().wait_stream(stream)
        old();cold.replay();runtime.exact(y0,y1)
        assert len(controller.owners)==1 and controller.fallbacks['cold_capture']==1
        # A registered static pack captured unexpectedly invalidates the
        # predicate and uses the full kernel. Later eager refresh restores it.
        stale=torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(stale,stream=stream):
            half.pack_half_weight(master,out=weight)
            new()
        torch.cuda.current_stream().wait_stream(stream)
        assert not owner.valid and controller.fallbacks['stale_capture']==1
        master.add_(.00013);runtime.ORIGINAL_PACK(master,out=weight);old();stale.replay();runtime.exact(y0,y1)
        half.pack_half_weight(master,out=weight);assert owner.valid and owner.count.data_ptr()==ptr
        assert torch.equal(rng,torch.cuda.get_rng_state())
        report=controller.report();assert report['owners']==1 and report['Count_bytes']==8
        print('DECISION_ZERO_LOW_PRODUCTION '+json.dumps(dict(fixture=fixture,history=history,
            original_preexisting_parts=True,refresh_while_mode_disabled=True,Count_same_pointer=True,
            general_Half_zero_restore_bits=True,cold_capture_fallback=True,stale_capture_fallback=True,
            eager_refresh_restores=True,RNG_unchanged=True,report=report)),flush=True)
        for g in [graph,cold,stale]:g.reset()
        clear_fixture()
        assert half.pack_half_weight is runtime.packed and prepacked._prepacked_gemm is runtime.ORIGINAL_JIT and runtime.ORIGINAL_JIT.run is runtime.launched
    print('DECISION_ZERO_LOW_PRODUCTION_CHECK passed=true fixtures=2',flush=True)


if __name__=='__main__':main()
