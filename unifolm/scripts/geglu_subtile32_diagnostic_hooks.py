"""Diagnostic-only interception of the two accepted world GEGLU producers."""
from contextlib import contextmanager
import os
import torch
from unifolm_wma.utils import geglu_hidden_parts_pipeline as c640
from unifolm_wma.utils import geglu_hidden_parts_c320_pipeline as c320
from unifolm_wma.utils.triton_geglu_subtile32 import _geglu_subtile32

MODE='UNIFOLM_GEGLU_SUBTILE32_DIAGNOSTIC'
GEOMETRIES={(40960,1280,320),(10240,2560,640)}


@contextmanager
def installed():
    originals={module:module._zero_geglu_hidden_parts for module in [c320,c640]}
    launches=[]
    class Observed:
        def __init__(self,original):self.original=original
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                mode=os.environ.get(MODE,'0');assert mode in {'0','1'}
                geometry=(args[10],args[11],args[12])
                selected=(mode=='1' and not torch.backends.cudnn.allow_tf32 and
                          not torch.backends.cuda.matmul.allow_tf32 and geometry in GEOMETRIES)
                if selected:
                    assert args[14:17]==(64,64,64) and not args[18] and args[19]
                    assert kwargs['num_warps']==4 and kwargs['num_stages']==2 and not kwargs['enable_fp_fusion']
                launches.append(dict(M=args[10],C=args[11],K=args[12],group=args[17],subtile32=selected))
                return (_geglu_subtile32 if selected else self.original)[grid](*args,**kwargs)
            return call
    for module,original in originals.items():module._zero_geglu_hidden_parts=Observed(original)
    try:yield launches
    finally:
        for module,original in originals.items():module._zero_geglu_hidden_parts=original
