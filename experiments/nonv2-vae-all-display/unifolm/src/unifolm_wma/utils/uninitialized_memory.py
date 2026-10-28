"""Explicit allocation-fill policy, independent of deterministic algorithms."""
import os
import torch
from torch.utils import deterministic


def configure_empty_fill():
    mode = os.environ.get('UNIFOLM_FILL_UNINITIALIZED_MEMORY', '1')
    if mode not in {'0', '1'}:
        raise ValueError('UNIFOLM_FILL_UNINITIALIZED_MEMORY must be 0 or 1')
    deterministic.fill_uninitialized_memory = mode == '1'
    print('UNINITIALIZED_MEMORY_FILL enabled='+str(mode == '1').lower()+
          ' deterministic_algorithms='+str(torch.are_deterministic_algorithms_enabled()).lower(),
          flush=True)
