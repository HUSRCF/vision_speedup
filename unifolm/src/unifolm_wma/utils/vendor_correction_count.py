"""Persistent device predicate refreshed by either stage's original weight pack."""
import torch


def signature(parts):
    if len(parts)!=2 or any(p.shape!=(640,2560) or p.dtype!=torch.float16
            or not p.is_cuda or not p.is_contiguous() or p.device!=parts[0].device for p in parts):
        return None
    try:return tuple((p.data_ptr(),p._version,tuple(p.shape),p.dtype,p.device) for p in parts)
    except RuntimeError:return None


def get_count(parts):
    key=signature(parts)
    if key is None:return None
    cached=getattr(parts[0],'_decision_vendor_correction_count',None)
    if cached is None or cached[0]!=key:return None
    count=cached[1]
    if count.shape!=() or count.dtype!=torch.int64 or count.device!=parts[0].device:return None
    return count


def refresh_count(parts):
    key=signature(parts)
    if key is None or torch.cuda.is_current_stream_capturing():return None
    cached=getattr(parts[0],'_decision_vendor_correction_count',None)
    updated=torch.count_nonzero(parts[1])
    if cached is not None and cached[1].shape==() and cached[1].dtype==torch.int64 and cached[1].device==updated.device:
        cached[1].copy_(updated);updated=cached[1]
    parts[0]._decision_vendor_correction_count=(key,updated)
    return updated
