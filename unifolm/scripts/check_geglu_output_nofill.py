"""Prove complete GEGLU output stores for each production branch and tail."""
import json
import os
from unittest.mock import patch
import torch
from torch.utils import deterministic
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils import geglu_output_allocation as allocation
from unifolm_wma.utils.triton_primary_narrow_geglu import primary_narrow_geglu as decision320
from unifolm_wma.utils.triton_world_narrow_geglu import primary_narrow_geglu as world320
from unifolm_wma.utils.triton_primary_geglu import primary_geglu as decision_wide
from unifolm_wma.utils.triton_branch_protected_geglu import branch_protected_geglu as world640
from unifolm_wma.utils.triton_prepacked_geglu import prepacked_geglu as world1280


@torch.no_grad()
def main():
    torch.manual_seed(881)
    torch.use_deterministic_algorithms(True)
    assert deterministic.fill_uninitialized_memory
    options=[('decision320',decision320,dict(mode='both',block_n=64),40960,320),
             ('world320',world320,dict(mode='gate',block_n=64),40960,320),
             ('decision640',decision_wide,dict(mode='both'),10240,640),
             ('world640',world640,dict(mode='gate',block_n=64),10240,640),
             ('decision1280large',decision_wide,dict(mode='both'),2560,1280),
             ('decision1280small',decision_wide,dict(mode='both'),640,1280),
             ('world1280large',world1280,dict(block_n=32),2560,1280),
             ('world1280small',world1280,dict(block_n=32),640,1280)]
    fixtures=[(name,fn,kw,rows,inner,4*inner,True) for name,fn,kw,rows,inner in options]
    fixtures += [(name+'-tails',fn,kw,73,97,137,bias) for name,fn,kw,_,_ in
                 [options[i] for i in [0,1,2,3,6]] for bias in [True,False]]
    cases=0
    for name,fn,kwargs,rows,inner,hidden,bias_on in fixtures:
        value=torch.randn(rows,inner,device='cuda')*.7
        weight=torch.randn(2*hidden,inner,device='cuda')*.02
        parts=pack_half_weight(weight)
        bias=torch.randn(2*hidden,device='cuda')*.03 if bias_on else None
        def run(x,mode):
            os.environ['UNIFOLM_GEGLU_OUTPUT_NOFILL']=mode
            return fn(x,parts,bias,**kwargs)
        baseline=run(value,'0')
        allocation._debug_poison=float('nan')
        actual=run(value,'1')
        assert torch.equal(actual.view(torch.int32),baseline.view(torch.int32)),name
        allocation._debug_poison=None
        before=graph_seconds(lambda x:run(x,'0'),value,5,3)
        seconds=graph_seconds(lambda x:run(x,'1'),value,5,3)
        after=graph_seconds(lambda x:run(x,'0'),value,5,3)
        os.environ['UNIFOLM_GEGLU_OUTPUT_NOFILL']='1'
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):run(value,'1')
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=run(value,'1')
        value.add_(.0001)
        weight.mul_(.999)
        pack_half_weight(weight,out=parts)
        if bias is not None:bias.mul_(.999)
        reference=run(value,'0')
        graph.replay()
        assert torch.equal(captured.view(torch.int32),reference.view(torch.int32)),name
        assert deterministic.fill_uninitialized_memory
        assert torch.isnan(torch.empty_like(value)).all()
        cases+=1
        print('GEGLU_OUTPUT_NOFILL '+json.dumps(dict(name=name,shape=[rows,inner,hidden],
            bias=bias_on,before_s=before,after_s=after,candidate_s=seconds,
            conservative_speedup=min(before,after)/seconds)),flush=True)
    os.environ['UNIFOLM_GEGLU_OUTPUT_NOFILL']='1'
    with patch.object(torch,'empty',side_effect=RuntimeError('Injected allocation failure')):
        try:allocation.empty((1,),device='cuda')
        except RuntimeError:pass
        else:raise AssertionError('Allocation exception lost')
    assert deterministic.fill_uninitialized_memory
    print(f'GEGLU_OUTPUT_NOFILL_CHECK passed=true cases={cases} helpers=5 '
          'actual_stage_branches=true all_output_signed_zero_bitwise=true full_nan_poison=true '
          'odd_mkn_bias_branches=true input_master_weight_bias_graph_refresh=true '
          'native_fill_retained=true allocation_exception_restore=true full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
