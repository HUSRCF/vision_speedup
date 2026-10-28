"""Whole-call timings include K/V packing; require exact original output bits."""
import json
import os
import torch
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_compensated_attention import compensated_attention
from unifolm_wma.utils.triton_prepacked_compensated_attention import prepacked_compensated_attention
from unifolm_wma.utils import attention_buffer_allocation as allocation


def exact(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32)),float((actual-expected).abs().max())


@torch.no_grad()
def main():
    torch.manual_seed(947)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ['UNIFOLM_ATTENTION_OUTPUT_NOFILL']='all'
    os.environ['UNIFOLM_HALF_PACK_NOFILL']='1'
    for index,(batch,rows,keys,heads) in enumerate([
            (16,2560,2560,5),(16,2560,2560,8),(16,640,640,10),
            (16,160,160,20),(2,73,97,3),(1,1,3,1)]):
        q=torch.randn(batch,rows,heads,64,device='cuda')*.5
        k=torch.randn(batch,keys,heads,64,device='cuda')*.5
        v=torch.randn_like(k)*.5
        kwargs=dict(block_m=64,block_n=64,num_warps=4)
        def baseline(x):
            os.environ['UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV']='0'
            return compensated_attention(x,k,v,**kwargs)
        def candidate(x):
            os.environ['UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV']='kv'
            return compensated_attention(x,k,v,**kwargs)
        reference=baseline(q)
        allocation._debug_poison=float('nan')
        actual=candidate(q)
        allocation._debug_poison=None
        assert torch.isfinite(actual).all();exact(actual,reference)
        before=graph_seconds(baseline,q,10,5)
        seconds=graph_seconds(candidate,q,10,5)
        after=graph_seconds(baseline,q,10,5)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):candidate(q)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=candidate(q)
        q.add_(.001);k.mul_(.997);v.sub_(.002)
        graph.replay();exact(captured,baseline(q))
        print('PREPACKED_ATTENTION_RUNTIME '+json.dumps(dict(index=index,
            shape=[batch,rows,keys,heads],before_s=before,candidate_s=seconds,
            after_s=after,conservative_speedup=min(before,after)/seconds,
            exact_bits=True,graph_changed_qkv=True,packing_included=True)),flush=True)
        del q,k,v,reference,actual,captured,graph
        torch.cuda.empty_cache()
    print('PREPACKED_ATTENTION_RUNTIME_CHECK passed=true cases=6 '
          'signed_zero_bits=true odd_mn=true poison=true graph_changed_qkv=true '
          'packing_included=true full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
