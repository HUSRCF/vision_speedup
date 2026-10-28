"""Original GE count/cache refresh without the world-only policy gate."""
import torch

def _refresh_shared_ge_count(parts,*,refresh=False):
    try:
        signature=tuple((p.data_ptr(),p._version,tuple(p.shape),p.dtype,p.device) for p in parts)
    except RuntimeError:return None
    cached=getattr(parts[0],'_geglu_zero_weight_low_count',None)
    if not refresh and cached is not None and cached[0]==signature:return cached[1]
    if torch.cuda.is_current_stream_capturing():return None
    count=torch.count_nonzero(parts[1])
    if cached is not None and cached[1].device==count.device:
        cached[1].copy_(count);count=cached[1]
    parts[0]._geglu_zero_weight_low_count=(signature,count)
    return count

