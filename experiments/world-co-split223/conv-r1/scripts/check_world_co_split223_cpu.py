"""Check the exact channel domain and full layout restoration without CUDA."""
import json,os
from pathlib import Path
import torch
import torch.distributed as dist
from world_co_split223 import domain,gather_channels
specs=json.loads((Path(os.environ['EXPERIMENT_ROOT'])/'geometries.json').read_text())
checks=0
for spec in specs:
    co,cols,k=spec['geometry'];bm,bn,bk,_,_=spec['config']
    visited=[]
    for rank in(0,1):
        d=domain(spec['geometry'],spec['config'],rank)
        rows=[d['row_start']+tile*bm+lane for tile in range((co//2)//bm)for lane in range(bm)]
        assert rows==list(range(rank*(co//2),(rank+1)*(co//2)));visited+=rows;checks+=1
    assert visited==list(range(co))
old=dist.all_gather_into_tensor
try:
    for c in(320,640,1280):
        raw=torch.arange(16*c*2*3,dtype=torch.float32).view(16,c,2,3)
        for kind in('NCHW','NC_permuted','NHWC'):
            x=raw.contiguous()if kind=='NCHW'else raw.permute(1,0,2,3).contiguous().permute(1,0,2,3)if kind=='NC_permuted'else raw.contiguous(memory_format=torch.channels_last)
            for rank in(0,1):
                def gather(out,local):
                    assert torch.equal(local,x[:,rank*(c//2):(rank+1)*(c//2)])
                    out[:16].copy_(x[:,:c//2]);out[16:].copy_(x[:,c//2:])
                dist.all_gather_into_tensor=gather
                y=gather_channels(x,rank);assert y.stride()==x.stride()and torch.equal(y.contiguous().view(torch.uint8),x.contiguous().view(torch.uint8));checks+=1
finally:dist.all_gather_into_tensor=old
assert not torch.cuda.is_initialized()
print('WORLD_CO_SPLIT_CPU '+json.dumps(dict(passed=True,checks=checks,geometries=len(specs),complete_channel_domain=True,full_layout_restore=True,no_CUDA_initialized=True)))
