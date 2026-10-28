"""Observe the actual two launches after the production host chooses policy."""
from contextlib import contextmanager
import os
from unittest.mock import patch
import torch
import unifolm_wma.utils.partitioned_temporal as module


@contextmanager
def installed():
    launches=[]
    original=module._partitioned_temporal
    class Observed:
        def __getitem__(self,grid):
            run=original[grid]
            def launch(*args,**kwargs):
                p=dict(zip(original.arg_names,args));p.update(kwargs)
                c=p['C'];default_group=8 if c==320 else 16
                eligible=(p['M'],p['N'],p['C'],p['T'],p['S']) in {(40960,320,320,16,2560),(10240,640,640,16,640)}
                # These observers are used only on the default production tiles.
                if eligible:
                    tuned=(os.environ.get('UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE','2')=='3'
                        and not p['THREE'] and not torch.backends.cudnn.allow_tf32
                        and not torch.backends.cuda.matmul.allow_tf32)
                    assert p['GROUP_M']==(32 if tuned and c==320 else default_group)
                    assert kwargs['num_stages']==(3 if tuned else 2)
                assert (p['BM'],p['BN'],p['BK'],kwargs['num_warps'])==(64,64,64,4)
                assert kwargs['enable_fp_fusion'] is False
                launches.append(dict(M=p['M'],N=p['N'],C=c,T=p['T'],S=p['S'],three=p['THREE'],
                                     group=p['GROUP_M'],stages=kwargs['num_stages'],count_ptr=p['LowCount'].data_ptr()))
                return run(*args,**kwargs)
            return launch
    with patch.object(module,'_partitioned_temporal',Observed()):
        yield launches
