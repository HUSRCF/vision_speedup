"""Check FP64 samples, tail masks, graph refresh and actual attention shapes."""
import json
import torch
import xformers.ops
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_compensated_attention import compensated_attention


def sampled_error(query,key,value,actual):
    rows=sorted({0,min(1,query.shape[1]-1),query.shape[1]//2,query.shape[1]-1})
    errors=[]
    for batch in sorted({0,query.shape[0]-1}):
        for head in sorted({0,query.shape[2]-1}):
            q=query[batch,rows,head].double()
            k=key[batch,:,head].double()
            v=value[batch,:,head].double()
            precise=(q@k.t()*0.125).softmax(-1)@v
            observed=actual[batch,rows,head].double()
            errors.append(float((observed-precise).norm()/precise.norm()))
    return max(errors)


@torch.no_grad()
def compare(query,key,value,label):
    baseline=lambda q:xformers.ops.memory_efficient_attention(q,key,value)
    reference=baseline(query)
    before=graph_seconds(baseline,query,10,3)
    records=[]
    for bm,bn in ((32,64),(64,32),(32,32),(64,64)):
        proposed=lambda q:compensated_attention(q,key,value,block_m=bm,block_n=bn)
        actual,kernel=compensated_attention(query,key,value,block_m=bm,block_n=bn,
                                            return_kernel=True)
        assert actual.dtype==torch.float32 and torch.isfinite(actual).all()
        error=sampled_error(query,key,value,actual)
        delta=float((actual-reference).norm()/reference.norm())
        assert error<1e-5,(query.shape,bm,bn,error)
        after=graph_seconds(proposed,query,10,3)
        record=dict(label=label,q_shape=list(query.shape),k_shape=list(key.shape),
            config=[bm,bn],relative_to_fp64=error,relative_to_xformers=delta,
            native_s=before,candidate_s=after,speedup=before/after,
            registers=getattr(kernel,'n_regs',None),spills=getattr(kernel,'n_spills',None))
        records.append(record)
        print('COMPENSATED_ATTENTION '+json.dumps(record),flush=True)
    return records


@torch.no_grad()
def main():
    torch.manual_seed(181)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    # Unequal lengths exercise query and key tails without a full score tensor.
    q=torch.randn(2,35,3,64,device='cuda')*.7
    k=torch.randn(2,73,3,64,device='cuda')*.7
    v=torch.randn_like(k)
    output=compensated_attention(q,k,v)
    assert sampled_error(q,k,v,output)<1e-5
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):compensated_attention(q,k,v)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=compensated_attention(q,k,v)
    q.add_(.1);k.mul_(.9);v.sub_(.15)
    graph.replay()
    torch.testing.assert_close(captured,compensated_attention(q,k,v),rtol=0,atol=0)
    assert sampled_error(q,k,v,captured)<1e-5
    print('COMPENSATED_ATTENTION_MASK_GRAPH passed=true',flush=True)
    rows=[]
    for batch,length,heads in ((16,2560,5),(16,640,10),(16,160,20),(16,40,20)):
        q=torch.randn(batch,length,heads,64,device='cuda')*.7
        k=torch.randn_like(q)*.7
        v=torch.randn_like(q)
        rows.extend(compare(q,k,v,'synthetic-real-shape'))
        del q,k,v
        torch.cuda.empty_cache()
    print('COMPENSATED_ATTENTION_CHECK '+json.dumps(dict(
        passed=True,shapes=4,configs=4,full_video_quality='unverified',
        inference_installed=False)),flush=True)


if __name__=='__main__':main()
