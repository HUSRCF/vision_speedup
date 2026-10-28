"""CPU metadata guard for decision sharding; not a GPU numerical proof."""
from unittest.mock import patch
from contextlib import contextmanager
import torch
import torch.utils.checkpoint
import torch.distributed as dist
import nccl_ffn_209_mode as mode
from unifolm_wma.utils import vendor_correction_count as counts
from unifolm_wma.utils import paired_fused_vendor_hidden_pipeline as paired

@contextmanager
def decision_flag():
    before=torch.backends.cudnn.allow_tf32;torch.backends.cudnn.allow_tf32=True
    try:yield
    finally:torch.backends.cudnn.allow_tf32=before

def main():
    assert not torch.cuda.is_initialized()
    xh=torch.arange(16,dtype=torch.float16).view(8,2);xl=xh*.01
    residual=torch.zeros(8,3);bias=torch.zeros(3);weight=(torch.zeros(3,2,dtype=torch.float16),)*2
    seen=[]
    def producer(inputs,parts,bias):return tuple(v.clone() for v in inputs)
    def wrong_consumer(*args,**kwargs):raise AssertionError('Original full-M eligibility would fall back after row slicing')
    def final(stack,parts,value,count,bias,*,chunk_k,debug_poison):
        assert stack.shape==(2,4,2) and value.shape==(4,3) and chunk_k==512 and not debug_poison
        assert count.dtype==torch.int64 and int(count)==7;seen.append(True)
        return value+1
    def gather(output,local):output.copy_(torch.cat((local,local),dim=0))
    original=mode.GEGLU.run
    with patch.object(dist,'get_world_size',return_value=2),patch.object(dist,'get_rank',return_value=0),patch.object(dist,'all_gather_into_tensor',side_effect=gather),patch.object(torch.cuda,'is_current_stream_capturing',return_value=False),patch.object(mode.DECISION640,'primary_geglu_hidden_parts',side_effect=producer),patch.object(mode.DECISION640,'chunked_hidden_residual',side_effect=wrong_consumer),patch.object(counts,'get_count',return_value=torch.tensor(7,dtype=torch.int64)),patch.object(paired,'paired_hidden_residual',side_effect=final),decision_flag():
        controller=mode.install(torch.nn.Module())
        try:
            with mode.mode(True):
                hidden=mode.DECISION640.primary_geglu_hidden_parts((xh,xl),weight,bias)
                assert hidden[0].shape==(4,2)
                result=mode.DECISION640.chunked_hidden_residual(hidden,weight,residual,bias,chunk_k=512)
                assert result.shape==(8,3) and torch.equal(result,torch.ones_like(result))
            assert not controller.pending and len(seen)==1
        finally:mode.uninstall()
    assert mode.GEGLU.run==original and not torch.cuda.is_initialized()
    print('NCCL_FFN_209_CPU_CHECK passed=true paired_consumer_preserved=true chunk512=true no_GPU_math_claim=true',flush=True)

if __name__=='__main__':main()
