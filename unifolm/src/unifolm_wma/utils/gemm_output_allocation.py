"""Local prefill policy for fully overwritten custom GEMM output tensors."""
import os
import torch
from torch.utils import deterministic

_debug_poison=None


def empty(kind,*args,**kwargs):
    mode=os.environ.get('UNIFOLM_GEMM_OUTPUT_NOFILL','0')
    if kind not in {'classic','primary'} or mode not in {'0','classic','primary','all'}:
        raise ValueError('Invalid GEMM output allocation scope')
    if mode not in {kind,'all'}:return torch.empty(*args,**kwargs)
    previous=deterministic.fill_uninitialized_memory
    try:
        deterministic.fill_uninitialized_memory=False
        result=torch.empty(*args,**kwargs)
    finally:
        deterministic.fill_uninitialized_memory=previous
    if _debug_poison is not None:result.fill_(_debug_poison)
    return result
