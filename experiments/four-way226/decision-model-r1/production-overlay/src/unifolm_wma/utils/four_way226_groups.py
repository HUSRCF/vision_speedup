"""Select the declared four-way or native-pair collective by tensor extent."""
import torch.distributed as dist
from contextlib import contextmanager
import os
from pathlib import Path

parent=dist.all_gather_into_tensor
pair=None
groups=[]
calls={2:0,4:0}

@contextmanager
def kernel_cache(rank):
    assert 'TRITON_CACHE_MANAGER' not in os.environ
    directory=Path(os.environ['FOUR_WAY_KERNEL_CACHE_DIR'])/('rank-'+str(rank))
    directory.mkdir(mode=0o700,exist_ok=True)
    before=os.environ.get('TRITON_CACHE_DIR');os.environ['TRITON_CACHE_DIR']=str(directory)
    try:yield
    finally:
        if before is None:os.environ.pop('TRITON_CACHE_DIR',None)
        else:os.environ['TRITON_CACHE_DIR']=before

def degree(output,local):
    assert output.dtype==local.dtype and output.device==local.device
    assert output.numel()%local.numel()==0
    count=output.numel()//local.numel()
    assert count in (2,4)
    return count

def gather(output,local,group=None,async_op=False):
    assert group is None and not async_op and pair is not None
    count=degree(output,local);calls[count]+=1
    return parent(output,local,group=pair if count==2 else None,async_op=False)

def initialize():
    global pair,groups
    assert dist.is_initialized() and dist.get_world_size()==4 and pair is None
    assert dist.all_gather_into_tensor is parent
    for ranks in ((0,1),(2,3)):
        group=dist.new_group(list(ranks),backend='nccl');groups.append(group)
        if dist.get_rank() in ranks:pair=group
    assert pair is not None
    dist.all_gather_into_tensor=gather

def close():
    global pair,groups
    assert dist.all_gather_into_tensor is gather
    dist.all_gather_into_tensor=parent
    for group in groups:
        if group!=dist.GroupMember.NON_GROUP_MEMBER:dist.destroy_process_group(group)
    pair=None;groups=[]
