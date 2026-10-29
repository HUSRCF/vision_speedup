import torch
from nccl_root_model_move import collect,buckets,layout
from audit_nccl_root_model_move import Fixture
assert not torch.cuda.is_initialized()
m=Fixture();entries=collect(m)
assert len(entries)==8 and len(m.state_dict())==10
assert sum(len(e['slots'])for e in entries)==10
for limit in (32,4096,1024*1024):
 chunks=list(buckets(entries,limit));seen=[id(e['tensor'])for _,chunk in chunks for e in chunk]
 assert len(seen)==len(set(seen))==8
 assert sum(e['tensor'].numel()*e['tensor'].element_size()for _,chunk in chunks for e in chunk)==sum(t.numel()*t.element_size()for t in [e['tensor']for e in entries])
 for dtype,chunk in chunks:
  spans,total=layout(chunk);last=0
  for e,offset in spans:
   assert offset>=last and (offset*e['tensor'].element_size())%256==0
   last=offset+e['tensor'].numel()
  assert total==last
assert not torch.cuda.is_initialized()
print('ROOT_MODEL_MOVE_CPU passed=true alias_slots=true dtype_buckets=true no_GPU=true')
