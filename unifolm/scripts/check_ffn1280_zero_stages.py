"""Original eight both-stage guards plus explicit dispatch-policy checks."""
import json
import os
import torch
from check_both_c1280_zero_parts import main as original_guards
from unifolm_wma.utils import ffn1280_zero_residual as consumer
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight


@torch.no_grad()
def main():
    assert os.environ['UNIFOLM_WORLD_FFN1280_ZERO_STAGES']=='3'
    original=consumer._zero_low_gemm_residual
    launches=[]
    class Observed:
        def __getitem__(self,grid):
            launch=original[grid]
            def call(*args,**kwargs):
                launches.append((args[8],args[9],args[10],kwargs['num_stages']))
                assert kwargs['num_warps']==4
                return launch(*args,**kwargs)
            return call
    consumer._zero_low_gemm_residual=Observed()
    try:
        original_guards()
        assert launches and all((m,n,k) in {(2560,1280,5120),(640,1280,5120)} and d==3 for m,n,k,d in launches)
        eligible=len(launches)
        x=pack_half_weight(torch.randn(13,5120,device='cuda')*.1,weight_scale=1.)
        w=pack_half_weight(torch.randn(71,5120,device='cuda')*.01)
        count=consumer.shared_final_low_count(w)
        residual=torch.randn(13,71,device='cuda')*.1
        output=consumer.zero_ffn1280_hidden_residual(x,w,count,residual)
        assert launches[-1]==(13,71,5120,2)
        os.environ['UNIFOLM_WORLD_FFN1280_ZERO_STAGES']='2'
        baseline=consumer.zero_ffn1280_hidden_residual(x,w,count,residual)
        assert launches[-1]==(13,71,5120,2)
        assert torch.equal(output.view(torch.int32),baseline.view(torch.int32))
        os.environ['UNIFOLM_WORLD_FFN1280_ZERO_STAGES']='invalid'
        try:
            consumer.zero_ffn1280_hidden_residual(x,w,count,residual)
        except ValueError as exc:
            assert str(exc)=='Invalid world C1280 FFN depth'
        else:
            raise AssertionError('invalid depth accepted')
    finally:
        os.environ['UNIFOLM_WORLD_FFN1280_ZERO_STAGES']='3'
        consumer._zero_low_gemm_residual=original
    print('FFN1280_ZERO_STAGES_RUNTIME_CHECK '+json.dumps(dict(passed=True,real_block_fixtures=8,
        eligible_launches=eligible,actual_shapes_depth3=True,other_shapes_depth2=True,
        explicit_depth2=True,invalid_flag_rejected=True)),flush=True)


if __name__=='__main__':main()
