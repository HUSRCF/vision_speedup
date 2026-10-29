"""Check native frame/mask dispatch and final gather without CUDA initialization."""
import torch
from nonv2_native_attention59_mode import local_attention
assert not torch.cuda.is_initialized();cases=0
for heads in(5,10,20):
 for query_tokens,key_tokens in[(40,40),(640,77),(2560,2),(40,16),(640,256)]:
  q=torch.arange(16,dtype=torch.float16)[:,None,None,None].expand(16,query_tokens,heads,64)
  k=torch.arange(16,dtype=torch.float16)[:,None,None,None].expand(16,key_tokens,heads,64);v=k.clone()
  mask=torch.arange(16)[:,None,None,None].expand(16,heads,query_tokens,key_tokens)
  for rank in(0,1):
   def parent(a,b,c,attn_bias,**kw):
    assert a.shape[0]==b.shape[0]==c.shape[0]==8
    expected=torch.arange(8*rank,8*(rank+1))
    assert torch.equal(a[:,0,0,0].long(),expected)and torch.equal(b[:,0,0,0].long(),expected)
    assert torch.equal(attn_bias[:,0,0,0],expected)
    return a.clone()
   result=local_attention(parent,q,k,v,mask,rank,p=0.,scale=None,op=None,output_dtype=None)
   assert result.shape==q.shape and torch.equal(result[rank*8:(rank+1)*8],q[rank*8:(rank+1)*8]);cases+=1
assert not torch.cuda.is_initialized()
print('NONV2_NATIVE_ATTENTION59_CPU passed=true cases='+str(cases)+' global_frame_head_mask=true no_GPU=true')
