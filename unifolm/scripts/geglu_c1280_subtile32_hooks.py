"""Observe the production host's kernel choice, including all fallback cases."""
from contextlib import contextmanager
import os
from unittest.mock import patch
import torch
import unifolm_wma.utils.geglu_hidden_parts_c1280_pipeline as module


@contextmanager
def installed():
    launches=[]
    def observe(original,new):
        class Observed:
            arg_names=original.arg_names
            def __getitem__(self,grid):
                run=original[grid]
                def launch(*args,**kwargs):
                    p=dict(zip(original.arg_names,args));p.update(kwargs)
                    selected=(os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32','0')=='1'
                        and (p['M'],p['C'],p['K']) in {(2560,5120,1280),(640,5120,1280)}
                        and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
                    assert selected==new
                    assert (p['BM'],p['BN'],p['BK'],p['GROUP'],kwargs['num_warps'])==(64,32,64,32,4)
                    assert not p['PRIMARY_VALUE'] and not p['PRIMARY_GATE'] and kwargs['enable_fp_fusion'] is False
                    assert kwargs['num_stages']==(2 if new else int(os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_STAGES','2')))
                    launches.append(dict(M=p['M'],C=p['C'],K=p['K'],subtile32=new,stages=kwargs['num_stages']))
                    return run(*args,**kwargs)
                return launch
        return Observed()
    with patch.object(module,'_zero_geglu_hidden_parts',observe(module._zero_geglu_hidden_parts,False)):
        with patch.object(module,'_both_geglu_subtile32',observe(module._both_geglu_subtile32,True)):
            yield launches
