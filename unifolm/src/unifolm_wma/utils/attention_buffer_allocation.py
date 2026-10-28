"""Scoped allocation for outputs completely overwritten by attention kernels."""
import os
import torch
from torch.utils import deterministic

_debug_poison = None


def empty_like(kind, value):
    mode = os.environ.get('UNIFOLM_ATTENTION_OUTPUT_NOFILL', '0')
    if kind not in {'spatial', 'temporal'} or mode not in {'0', 'spatial', 'temporal', 'all'}:
        raise ValueError('Invalid attention output allocation scope')
    if mode not in {kind, 'all'}:
        return torch.empty_like(value)
    previous = deterministic.fill_uninitialized_memory
    try:
        deterministic.fill_uninitialized_memory = False
        result = torch.empty_like(value)
    finally:
        deterministic.fill_uninitialized_memory = previous
    if _debug_poison is not None:
        result.fill_(_debug_poison)
    return result
