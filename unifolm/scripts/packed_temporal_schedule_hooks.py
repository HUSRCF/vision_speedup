"""Observe packed kernel geometry and cache/predicate identity directly."""
from contextlib import contextmanager
import os
from unittest.mock import patch
import torch
import unifolm_wma.utils.triton_zero_weight_low_temporal as module


@contextmanager
def installed():
    original=module._zero_low_temporal;launches=[]
    class Observed:
        arg_names=original.arg_names
        def __getitem__(self,grid):
            run=original[grid]
            def launch(*args,**kwargs):
                p=dict(zip(original.arg_names,args));p.update(kwargs)
                if not p['IMPLICIT']:
                    selected=(os.environ.get('UNIFOLM_WORLD_PACKED_TEMPORAL_SCHEDULE')=='1'
                        and (p['M'],p['N'],p['C'],p['T'],p['S']) in {(2560,1280,1280,16,160),(640,1280,1280,16,40)}
                        and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
                    expected_group=(8 if p['S']==160 else 32) if selected else 32
                    assert p['GROUP']==expected_group
                    assert kwargs['num_stages']==(3 if selected else 2) and kwargs['num_warps']==4
                    assert kwargs['enable_fp_fusion'] is False
                    assert p['LowCount'].shape==() and p['LowCount'].dtype==torch.int64
                    launches.append(dict(M=p['M'],N=p['N'],C=p['C'],T=p['T'],S=p['S'],group=p['GROUP'],stages=kwargs['num_stages'],
                        selected=selected,count_ptr=p['LowCount'].data_ptr(),WH_ptr=p['WH'].data_ptr(),WL_ptr=p['WL'].data_ptr(),
                        count_nonzero=None if torch.cuda.is_current_stream_capturing() else int(p['LowCount'])))
                return run(*args,**kwargs)
            return launch
    with patch.object(module,'_zero_low_temporal',Observed()):yield launches
