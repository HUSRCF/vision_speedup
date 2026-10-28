"""Observe the actual host branch without selecting a shader on its behalf."""
from contextlib import contextmanager
import os
import torch
from unifolm_wma.utils import geglu_hidden_parts_pipeline as c640
from unifolm_wma.utils import geglu_hidden_parts_c320_pipeline as c320

MODE='UNIFOLM_WORLD_GEGLU_SUBTILE32'
GEOMETRIES={(40960,1280,320),(10240,2560,640)}


@contextmanager
def installed():
    originals={(module,name):getattr(module,name) for module in [c320,c640]
               for name in ['_zero_geglu_hidden_parts','_geglu_subtile32']}
    launches=[]
    class Observed:
        def __init__(self,original,selected):self.original=original;self.selected=selected
        def __getitem__(self,grid):
            launch=self.original[grid]
            def call(*args,**kwargs):
                mode=os.environ.get(MODE,'0');assert mode in {'0','1'}
                geometry=(args[10],args[11],args[12])
                expected=(mode=='1' and geometry in GEOMETRIES and not torch.backends.cudnn.allow_tf32
                          and not torch.backends.cuda.matmul.allow_tf32)
                assert self.selected==expected
                assert kwargs['num_warps']==4 and kwargs['num_stages']==2 and not kwargs['enable_fp_fusion']
                assert args[14:17]==(64,64,64) and not args[18] and args[19]
                launches.append(dict(M=args[10],C=args[11],K=args[12],group=args[17],subtile32=self.selected))
                return launch(*args,**kwargs)
            return call
    for (module,name),original in originals.items():setattr(module,name,Observed(original,name=='_geglu_subtile32'))
    try:yield launches
    finally:
        for (module,name),original in originals.items():setattr(module,name,original)
