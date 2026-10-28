"""Observe the host-selected kernel parameters without changing them."""
from contextlib import contextmanager
import os
from unittest.mock import patch
import torch
import unifolm_wma.utils.attention_hidden_parts_pipeline as module


@contextmanager
def installed():
    original=module._packed_attention_output_gemm;launches=[]
    class Observed:
        arg_names=original.arg_names
        def __getitem__(self,grid):
            run=original[grid]
            def launch(*args,**kwargs):
                p=dict(zip(original.arg_names,args));p.update(kwargs)
                selected=(os.environ.get('UNIFOLM_WORLD_ATTENTION_PROJECTION_SCHEDULE','0')=='1'
                    and not torch.is_grad_enabled() and p['INPUT_PROTECTED'] and p['HAS_BIAS']
                    and (tuple(p['XH'].shape),p['N']) in {((16,2560,5,64),320),((16,640,10,64),640)}
                    and p['Bias'].dtype==torch.float32 and p['Bias'].device==p['XH'].device and p['Bias'].is_contiguous()
                    and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
                assert [p[n] for n in ['BM','BN','BK']]==[64,64,64]
                assert p['GROUP_M']==(8 if selected else 32)
                assert kwargs['num_stages']==(3 if selected else 2) and kwargs['num_warps']==4 and kwargs['enable_fp_fusion'] is False
                launches.append(dict(M=p['M'],N=p['N'],K=p['K'],HAS_BIAS=p['HAS_BIAS'],INPUT_PROTECTED=p['INPUT_PROTECTED'],
                    BM=p['BM'],BN=p['BN'],BK=p['BK'],group=p['GROUP_M'],stages=kwargs['num_stages'],selected=selected,
                    WH_ptr=p['High'].data_ptr(),WL_ptr=p['Low'].data_ptr()))
                return run(*args,**kwargs)
            return launch
    with patch.object(module,'_packed_attention_output_gemm',Observed()):yield launches
