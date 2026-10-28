"""Check current classic/primary GEMM stores without changing their arithmetic."""
import json
import os
from unittest.mock import patch
import torch
from torch.utils import deterministic
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight,half_compensated_linear
from unifolm_wma.utils.triton_prepacked_input_linear import prepacked_input_linear
from unifolm_wma.utils.triton_primary_narrow_linear import primary_narrow_linear
from unifolm_wma.utils.triton_primary_grouped_qkv import primary_grouped_qkv
from unifolm_wma.utils import gemm_output_allocation as allocation


def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.view(torch.int32),b.view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(937)
    torch.use_deterministic_algorithms(True)
    assert deterministic.fill_uninitialized_memory
    fixtures=[]
    for rows,inner,out,bias in [(40960,320,320,False),(40960,1280,320,True),
            (10240,640,640,False),(40960,512,512,False),(40960,2048,512,True)]:
        fixtures.append(('inline','classic',rows,inner,out,bias,dict(block_m=64,block_n=64,
            group_m=32,block_k=32 if out==512 else 64)))
    for rows,inner,out,groups in [(10240,640,1920,3),(2560,1280,3840,3),
            (640,1280,3840,3),(2560,5120,1280,1),(640,5120,1280,1),
            (2560,1280,1280,1)]:
        fixtures.append(('prepacked','classic',rows,inner,out,groups==1,
            dict(block_m=64 if rows>=2560 else 32,block_n=64,block_k=64,group_m=32,output_groups=groups)))
    for inner,out,bias in [(320,320,False),(320,320,True),(512,512,False)]:
        fixtures.append(('primary','primary',40960,inner,out,bias,
            dict(tile=(64,128,32,4) if bias else (128,64,32,4))))
    fixtures.append(('qkv','primary',40960,320,320,False,dict(tile=(128,128,32,4,3))))
    for tag,kind in [('inline','classic'),('prepacked','classic'),('primary','primary'),('qkv','primary')]:
        for bias in ([False] if tag=='qkv' else [True,False]):
            options=dict(block_m=64,group_m=32,block_k=64) if tag=='inline' else {}
            fixtures.append((tag,kind,73,97,59,bias,options))
    for index,(tag,kind,rows,inner,outputs,bias_on,kwargs) in enumerate(fixtures):
        value=torch.randn(rows,inner,device='cuda')*.1
        weight=torch.randn(outputs*(3 if tag=='qkv' else 1),inner,device='cuda')*.01
        parts=pack_half_weight(weight)
        bias=torch.randn(outputs,device='cuda')*.03 if bias_on else None
        def compute(x):
            if tag=='inline':return half_compensated_linear(x,weight,bias,weight_parts=parts,**kwargs)
            if tag=='prepacked':return prepacked_input_linear(x,weight,bias,weight_parts=parts,**kwargs)
            if tag=='primary':return primary_narrow_linear(x,parts[0],bias,**kwargs)
            return primary_grouped_qkv(x,parts[0],**kwargs)
        def run(x,mode):
            os.environ['UNIFOLM_GEMM_OUTPUT_NOFILL']=mode
            return compute(x)
        reference=run(value,'0')
        allocation._debug_poison=float('nan')
        actual=run(value,kind)
        allocation._debug_poison=None
        assert torch.isfinite(actual).all()
        exact(actual,reference)
        before=graph_seconds(lambda x:run(x,'0'),value,5,3)
        seconds=graph_seconds(lambda x:run(x,kind),value,5,3)
        after=graph_seconds(lambda x:run(x,'0'),value,5,3)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):run(value,kind)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=run(value,kind)
        value.add_(.0001);weight.mul_(.999);pack_half_weight(weight,out=parts)
        if bias is not None:bias.add_(.0001)
        expected=run(value,'0');graph.replay();exact(captured,expected)
        assert deterministic.fill_uninitialized_memory and torch.isnan(torch.empty_like(value)).all()
        print('GEMM_OUTPUT_NOFILL '+json.dumps(dict(index=index,kind=kind,helper=tag,
            shape=[rows,inner,outputs],bias=bias_on,options=kwargs,before_s=before,
            after_s=after,candidate_s=seconds,conservative_speedup=min(before,after)/seconds)),flush=True)
    os.environ['UNIFOLM_GEMM_OUTPUT_NOFILL']='all'
    with patch.object(torch,'empty',side_effect=RuntimeError('Injected allocation failure')):
        try:allocation.empty('classic',(1,),device='cuda')
        except RuntimeError:pass
        else:raise AssertionError('Allocation error lost')
    assert deterministic.fill_uninitialized_memory
    print(f'GEMM_OUTPUT_NOFILL_CHECK passed=true cases={len(fixtures)} helpers=4 '
          'actual_shapes_and_initial512=true grouped_qkv_and_three_plane_outputs=true '
          'full_nan_poison_signed_zero_bits=true odd_mkn_bias_branches=true '
          'graph_input_master_bias_refresh=true native_fill_retained=true '
          'allocation_exception_restore=true full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
