"""Metadata observer must not read/serialize operand values or mutate inputs."""
import json
from types import SimpleNamespace
import torch
from temporal220_observer import describe

def kernel():pass
jit=SimpleNamespace(arg_names=['X','Y','M','N','Unknown'],fn=kernel)
assert not torch.cuda.is_initialized();torch.manual_seed(989)
rows=[]
for dtype in (torch.float32,torch.float16):
    for contiguous in (False,True):
        x=torch.randn(4,8).to(dtype);x=x if contiguous else x.t();y=x.clone()
        rng=torch.get_rng_state().clone();before=x.clone();args=(x,y,x.shape[0],x.shape[1],'never_export_this_arg')
        kwargs=dict(grid=(2,3),num_warps=4,num_stages=3,enable_fp_fusion=False)
        row=describe(jit,args,kwargs);text=json.dumps(row)
        assert 'never_export_this_arg' not in text and str(x.data_ptr()) not in text
        assert row['tensors']['X']['shape']==list(x.shape) and row['tensors']['X']['stride']==list(x.stride())
        assert row['scalars']=={'M':x.shape[0],'N':x.shape[1]} and row['grid']==[2,3]
        assert torch.equal(x,before) and torch.equal(rng,torch.get_rng_state())
        rows.append(dict(dtype=str(dtype),contiguous=contiguous,passed=True))
assert not torch.cuda.is_initialized()
print('TEMPORAL220_OBSERVER_CPU '+json.dumps(dict(passed=True,cases=len(rows),rows=rows,no_GPU_initialized=True,no_operand_values_or_addresses=True)))
