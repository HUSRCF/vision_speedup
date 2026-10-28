"""Local allocation policy for audited, fully written custom conv buffers."""
import os
import torch
from torch.utils import deterministic

_debug_poison=None


def empty(kind,*args,**kwargs):
    mode=os.environ.get('UNIFOLM_CONV_BUFFER_NOFILL','0')
    if kind not in {'temporal','spatial','f2'} or mode not in {'0','temporal','spatial','f2','all'}:
        raise ValueError('Invalid custom convolution allocation scope')
    if mode not in {kind,'all'}:return torch.empty(*args,**kwargs)
    previous=deterministic.fill_uninitialized_memory
    try:
        deterministic.fill_uninitialized_memory=False
        value=torch.empty(*args,**kwargs)
    finally:
        deterministic.fill_uninitialized_memory=previous
    if _debug_poison is not None:value.fill_(_debug_poison)
    return value
