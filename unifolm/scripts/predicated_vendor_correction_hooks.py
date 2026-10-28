"""Observe the actual third-product JIT rather than a diagnostic replacement."""
from contextlib import contextmanager
import os
from unittest.mock import patch
import torch
import unifolm_wma.utils.triton_predicated_chunked_correction as module


@contextmanager
def installed():
    original=module._third_product;launches=[]
    class Observed:
        arg_names=original.arg_names
        def __getitem__(self,grid):
            run=original[grid]
            def launch(*args,**kwargs):
                p=dict(zip(original.arg_names,args));p.update(kwargs)
                assert os.environ.get('UNIFOLM_DECISION_VENDOR_CORRECTION')=='1'
                assert torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
                assert (p['M'],p['N'],p['K'],p['CHUNK'],p['BM'],p['BN'])==(10240,640,2560,512,64,64)
                assert kwargs['num_warps']==4 and kwargs['num_stages']==3 and kwargs['enable_fp_fusion'] is False
                assert p['Active'].dtype==torch.int64 and p['Active'].shape==()
                launches.append(dict(M=p['M'],N=p['N'],K=p['K'],chunk=p['CHUNK'],count_ptr=p['Active'].data_ptr(),
                    count_nonzero=None if torch.cuda.is_current_stream_capturing() else int(p['Active']),
                    device_predicate=True))
                return run(*args,**kwargs)
            return launch
    with patch.object(module,'_third_product',Observed()):yield launches
