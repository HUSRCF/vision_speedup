"""Allocation policy only for fully overwritten High/Low split buffers."""
import os
import torch
from torch.utils import deterministic

_debug_poison=None


def empty_like(weight):
    mode=os.environ.get('UNIFOLM_HALF_PACK_NOFILL','0')
    if mode not in {'0','1'}:raise ValueError('Half pack allocation mode must be0or1')
    if mode=='0':return torch.empty_like(weight,dtype=torch.float16)
    previous=deterministic.fill_uninitialized_memory
    try:
        deterministic.fill_uninitialized_memory=False
        value=torch.empty_like(weight,dtype=torch.float16)
    finally:
        deterministic.fill_uninitialized_memory=previous
    if _debug_poison is not None:value.fill_(_debug_poison)
    return value
