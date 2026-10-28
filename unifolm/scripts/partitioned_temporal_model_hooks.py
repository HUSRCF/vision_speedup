"""Observe production wrapper dispatch; baseline never invokes the candidate."""
from contextlib import contextmanager
from unittest.mock import patch
import unifolm_wma.utils.world_partitioned_temporal as host


@contextmanager
def installed():
    launches=[]
    original=host.partitioned_temporal_conv
    def observed(value,parts,count,bias=None,**kwargs):
        launches.append(dict(shape=list(value.shape),group=kwargs['group_m'],
                             count_ptr=count.data_ptr(),part_ptrs=[p.data_ptr() for p in parts]))
        return original(value,parts,count,bias,**kwargs)
    with patch.object(host,'partitioned_temporal_conv',observed):
        yield launches
