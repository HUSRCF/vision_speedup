"""Validate global frame/head mask ownership and exact local mix tile domain."""
import torch
from nonv2_all_blocks59_mode import sample_slice,action_bias_slice
assert not torch.cuda.is_initialized();cases=0
for heads in (5,10,20):
    for keys in (16,256):
        prefix=(torch.arange(16)+1)*(keys//16)
        mask=torch.arange(keys)[None,:]>=prefix[:,None]
        mask=mask[:,None,:].repeat_interleave(heads,dim=0).expand(-1,40,-1)
        context=torch.arange(16)[:,None,None].expand(16,77,3)
        for rank in(0,1):
            actual=action_bias_slice(mask,rank,heads).reshape(8,heads,40,keys)
            expected=mask.reshape(16,heads,40,keys)[8*rank:8*(rank+1)]
            assert torch.equal(actual,expected)and torch.equal(sample_slice(context,rank),context[8*rank:8*(rank+1)])
            if rank==1 and keys==256:
                assert int((~actual[0,0,0]).sum())==144
            cases+=1
for tokens,channels in((2560,320),(640,640),(160,1280),(40,1280)):
    local=8*tokens*channels;assert local%1024==0
    last=(local//1024-1)*1024+torch.arange(1024)
    assert torch.equal(last<local,last<2*local)and int(last[-1])==local-1
    cases+=1
assert not torch.cuda.is_initialized()
print('NONV2_ALL_BLOCKS59_CPU passed=true cases='+str(cases)+' global_frame_head_mask=true whole_local_mix_tiles=true no_GPU=true')
