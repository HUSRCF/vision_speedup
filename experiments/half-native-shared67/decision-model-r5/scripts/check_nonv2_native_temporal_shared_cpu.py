"""Check spatial-position partition, 16-token retention and temporal bias slices."""
import torch
from nonv2_native_ffn_grid59_temporal_mode import local_temporal_attention,local_temporal_arguments

assert not torch.cuda.is_initialized()
cases=0
for batch in (40,160,640,2560):
    for heads in (5,8,20):
        q=torch.arange(batch,dtype=torch.float32)[:,None,None,None].expand(batch,16,heads,64)
        for kind in ('none','broadcast','per_position'):
            bias=None if kind=='none' else torch.arange(batch if kind=='per_position' else 1)[:,None,None,None].expand(batch if kind=='per_position' else 1,heads,16,16)
            for rank in (0,1):
                start,stop=rank*(batch//2),(rank+1)*(batch//2)
                def parent(a,b,c,attn_bias,**options):
                    assert a.shape==(batch//2,16,heads,64)
                    assert torch.equal(a[:,0,0,0],torch.arange(start,stop,dtype=a.dtype))
                    assert torch.equal(a,b) and torch.equal(b,c)
                    if kind=='per_position':assert torch.equal(attn_bias[:,0,0,0],torch.arange(start,stop))
                    elif kind=='broadcast':assert attn_bias.shape[0]==1
                    else:assert attn_bias is None
                    return a.clone()
                output=local_temporal_attention(parent,q,q,q,bias,rank,p=0.)
                assert output.shape==q.shape and torch.equal(output[start:stop],q[start:stop])
                cases+=1
assert cases==72 and not torch.cuda.is_initialized()
for batch in (40,160,640,2560):
    for heads in (5,8,20):
        q=torch.arange(batch,dtype=torch.float32)[:,None,None].expand(batch,16,heads*64).contiguous()
        out=torch.full_like(q,-1.)
        for rank in (0,1):
            a,b,c,y=local_temporal_arguments(q,q,q,out,rank)
            assert a.shape==(batch//2,16,heads*64) and torch.equal(a,b) and torch.equal(b,c)
            y.copy_(a)
        assert torch.equal(out,q)
assert not torch.cuda.is_initialized()
print('NONV2_NATIVE_TEMPORAL_SHARED_CPU passed=true cases=72 native_row_view_cases=24 full16_tokens=true global_position_bias=true no_GPU=true')
