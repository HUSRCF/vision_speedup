"""Skip prefill only for audited, fully written custom GEGLU outputs."""
import os
import torch
from torch.utils import deterministic

_debug_poison=None


def empty(*args,**kwargs):
    mode=os.environ.get('UNIFOLM_GEGLU_OUTPUT_NOFILL','0')
    if mode not in {'0','1'}:raise ValueError('GEGLU output allocation mode must be0or1')
    if mode=='0':return torch.empty(*args,**kwargs)
    previous=deterministic.fill_uninitialized_memory
    try:
        deterministic.fill_uninitialized_memory=False
        value=torch.empty(*args,**kwargs)
    finally:
        deterministic.fill_uninitialized_memory=previous
    if _debug_poison is not None:value.fill_(_debug_poison)
    return value
