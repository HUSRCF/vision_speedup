"""Skip prefill only for completely overwritten custom normalization buffers."""
import os
import torch
from torch.utils import deterministic


_debug_poison = None


def _allocate(kind, function, *args, **kwargs):
    mode = os.environ.get('UNIFOLM_NORM_BUFFER_NOFILL', '0')
    if mode not in {'0', 'group', 'residual', 'all'}:
        raise ValueError('Invalid normalization allocation scope')
    if mode not in {kind, 'all'}:
        return function(*args, **kwargs)
    previous = deterministic.fill_uninitialized_memory
    try:
        deterministic.fill_uninitialized_memory = False
        value = function(*args, **kwargs)
    finally:
        deterministic.fill_uninitialized_memory = previous
    if _debug_poison is not None:
        value.fill_(_debug_poison)
    return value


def empty(kind, *args, **kwargs):
    return _allocate(kind, torch.empty, *args, **kwargs)


def empty_like(kind, *args, **kwargs):
    return _allocate(kind, torch.empty_like, *args, **kwargs)
