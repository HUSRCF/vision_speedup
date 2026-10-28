"""Validate fresh producer storage and observe the actual paired consumer JIT."""
from contextlib import contextmanager
import os
from unittest.mock import patch
import torch
import unifolm_wma.utils.decision_chunked_hidden_pipeline as producer
import unifolm_wma.utils.paired_fused_vendor_hidden_pipeline as consumer


@contextmanager
def installed():
    launches=[]
    def observe(original,kind):
        class Observed:
            arg_names=original.arg_names
            def __getitem__(self,grid):
                run=original[grid]
                def launch(*args,**kwargs):
                    p=dict(zip(original.arg_names,args));p.update(kwargs)
                    paired=(os.environ.get('UNIFOLM_DECISION_PAIRED_FUSED_VENDOR')=='1'
                        and os.environ.get('UNIFOLM_DECISION_VENDOR_CORRECTION')=='1'
                        and torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
                        and p['M']==10240)
                    if kind=='producer':
                        assert (p['C'],p['K'],p['BM'],p['BN'],p['GROUP_M'])==(2560,640,64,64,32)
                        assert p['PRIMARY_GATE'] and kwargs['num_stages']==2 and kwargs['num_warps']==4
                        stack=getattr(p['YH'],'_paired_vendor_stack',None)
                        if paired:
                            assert stack is not None and stack.shape==(2,p['M'],2560) and stack.is_contiguous()
                            assert stack.data_ptr()==p['YH'].data_ptr()
                            assert p['YL'].data_ptr()==stack.data_ptr()+p['M']*2560*2
                            assert p['YL'].untyped_storage().data_ptr()==stack.untyped_storage().data_ptr()
                        else:assert stack is None
                        launches.append(dict(kind=kind,M=p['M'],C=p['C'],K=p['K'],paired=paired,activation_copy=False))
                    else:
                        assert paired and (p['N'],p['K'],p['CHUNK'],p['BM'],p['BN'])==(640,2560,512,16,32)
                        assert p['XH'].shape==(2,10240,2560) and p['XH'].is_contiguous()
                        assert p['Partial'].shape==(5,20480,640) and p['Partial'].dtype==torch.float32
                        assert p['Active'].shape==() and p['Active'].dtype==torch.int64
                        assert p['BLOCK']==256
                        assert kwargs['num_stages']==2 and kwargs['num_warps']==4
                        launches.append(dict(kind=kind,M=p['M'],N=p['N'],K=p['K'],chunk=p['CHUNK'],
                            count_ptr=p['Active'].data_ptr(),device_predicate=True,
                            count_nonzero=None if torch.cuda.is_current_stream_capturing() else int(p['Active'])))
                    assert kwargs['enable_fp_fusion'] is False
                    return run(*args,**kwargs)
                return launch
        return Observed()
    with patch.object(producer,'_primary_geglu_hidden_parts',observe(producer._primary_geglu_hidden_parts,'producer')):
        with patch.object(consumer,'_finish',observe(consumer._finish,'consumer')):
            yield launches
