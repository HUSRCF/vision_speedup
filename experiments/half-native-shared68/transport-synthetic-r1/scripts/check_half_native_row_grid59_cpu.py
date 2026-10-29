import torch
from nccl_root_model_move import collect,buckets
from audit_nccl_root_model_move import Fixture
assert not torch.cuda.is_initialized()
m=Fixture();entries=collect(m)
assert len(entries)==8 and len(m.state_dict())==10
assert sum(len(e['slots'])for e in entries)==10
for limit in (32,4096,1024*1024):
 chunks=list(buckets(entries,limit));seen=[id(e['tensor'])for _,chunk in chunks for e in chunk]
 assert len(seen)==len(set(seen))==8
 assert sum(e['tensor'].numel()*e['tensor'].element_size()for _,chunk in chunks for e in chunk)==sum(t.numel()*t.element_size()for t in [e['tensor']for e in entries])
assert not torch.cuda.is_initialized()
print('ROOT_MODEL_MOVE_CPU passed=true alias_slots=true dtype_buckets=true no_GPU=true')
