"""Stable, refreshable dtype buffers for fixed inference conditioning."""
import torch


def cast_fixed_condition(cache, name, value, dtype):
    if value.dtype == dtype or torch.is_grad_enabled():
        return value.to(dtype)
    # A graph captured for one shape must keep its buffer when another stage
    # uses a different conditioning shape between replays.
    signature = (name, tuple(value.shape), tuple(value.stride()), dtype, value.device)
    source_key = (value.data_ptr(), value._version, value.dtype)
    entry = cache.get(signature)
    if entry is not None and entry['key'] == source_key:
        return entry['output']
    converted = value.to(dtype)
    if entry is None:
        output = converted
    else:
        output = entry['output']
        output.copy_(converted)
    cache[signature] = dict(key=source_key, source=value, output=output)
    return output
