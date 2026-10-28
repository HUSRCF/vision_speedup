"""CPU-only launch-policy restoration, warmup, metadata and native fallback guards."""
import json
from pathlib import Path
import torch
import temporal_compensated_209_mode as mode

def main():
    configs=json.loads(Path('temporal-compensated-209-selection.json').read_text());calls=[]
    original=mode.ORIGINAL_RUN
    def fake(*args,**kwargs):calls.append(kwargs);return 'original'
    mode.ORIGINAL_RUN=fake;mode.ORIGINAL_JIT.run=fake
    try:
        controller=mode.install(configs)
        with torch.no_grad():
            q=torch.ones(1,16,64)
            args=(q,q,q,q,q,q,1,.125,False)
            assert mode.launched(*args,grid=(1,),warmup=True,num_warps=4,enable_fp_fusion=False)=='original' and calls[-1]['warmup']
            assert mode.launched(*args,grid=(1,),num_warps=4,enable_fp_fusion=False)=='original'
            with mode.mode(True):
                assert mode.launched(*args,grid=(1,),num_warps=4,enable_fp_fusion=False)=='original'
                try:
                    with mode.mode(False):raise RuntimeError('fixture')
                except RuntimeError:pass
                assert mode.enabled
            assert not mode.enabled and mode.ORIGINAL_JIT.run is mode.launched
        mode.uninstall();assert mode.ORIGINAL_JIT.run is fake and mode.active is None
        for bad in (configs[:-1],[dict(c,warps=4) for c in configs],[configs[0]]*5):
            try:mode.Controller(bad)
            except AssertionError:pass
            else:raise AssertionError('bad configuration accepted')
        assert not torch.cuda.is_initialized()
    finally:
        mode.uninstall();mode.ORIGINAL_RUN=original;mode.ORIGINAL_JIT.run=original
    print('TEMPORAL_COMPENSATED_209_CPU_CHECK passed=true no_CUDA_context=true',flush=True)

if __name__=='__main__':main()
