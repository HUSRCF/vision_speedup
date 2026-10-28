"""Observe production-selected QKV traversal without selecting it in the hook."""
from contextlib import contextmanager
import os
from unittest.mock import patch
import torch
import unifolm_wma.utils.triton_zero_weight_low_linear as module


@contextmanager
def installed():
    original=module._zero_low_gemm
    launches=[]
    class Observed:
        arg_names=original.arg_names
        def __getitem__(self,grid):
            run=original[grid]
            def launch(*args,**kwargs):
                p=dict(zip(original.arg_names,args));p.update(kwargs)
                eligible=((p['M'],p['N'],p['K']) in {(40960,960,320),(10240,1920,640),(2560,3840,1280)}
                    and p['OUTPUT_GROUPS']==3 and not p['HAS_BIAS']
                    and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
                wanted=1 if os.environ.get('UNIFOLM_WORLD_ZERO_QKV_GROUP','32')=='1-large' and eligible else 32
                assert p['GROUP']==wanted and kwargs['num_warps']==4 and kwargs['enable_fp_fusion'] is False
                launches.append(dict(M=p['M'],N=p['N'],K=p['K'],group=p['GROUP'],stages=kwargs['num_stages'],
                                     output_groups=p['OUTPUT_GROUPS'],bias=p['HAS_BIAS']))
                return run(*args,**kwargs)
            return launch
    with patch.object(module,'_zero_low_gemm',Observed()):
        yield launches
