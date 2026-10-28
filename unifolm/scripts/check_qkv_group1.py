"""Retain original real QKV/master/cache guards and test traversal policy."""
import json
import os
import torch
from check_zero_qkv_schedule import main as previous_guard
from qkv_group1_model_hooks import installed
from unifolm_wma.utils import triton_zero_weight_low_linear as helper
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight


def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
    assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    os.environ['UNIFOLM_WORLD_ZERO_QKV_GROUP']='1-large'
    with installed() as launches:
        previous_guard()
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        x=torch.randn(2560,1280,device='cuda')*.2
        w=(torch.randn(3840,1280,device='cuda')*.02).half().float()
        parts=pack_half_weight(w);count=helper.weight_low_count(parts)
        bias=torch.randn(3840,device='cuda')*.02
        for policy in ['selected','disabled','decision','matmul_tf32','bias','groups1','tail','small']:
            torch.backends.cudnn.allow_tf32=policy=='decision'
            torch.backends.cuda.matmul.allow_tf32=policy=='matmul_tf32'
            value=x[:2].contiguous() if policy=='tail' else x[:640].contiguous() if policy=='small' else x
            b=bias if policy=='bias' else None;groups=1 if policy=='groups1' else 3
            os.environ['UNIFOLM_WORLD_ZERO_QKV_GROUP']='32'
            expected=helper.zero_weight_low_linear(value,parts,count,b,output_groups=groups)
            os.environ['UNIFOLM_WORLD_ZERO_QKV_GROUP']='32' if policy=='disabled' else '1-large'
            before=len(launches);actual=helper.zero_weight_low_linear(value,parts,count,b,output_groups=groups)
            exact(actual,expected);assert len(launches)==before+1
            assert launches[-1]['group']==(1 if policy=='selected' else 32)
        os.environ['UNIFOLM_WORLD_ZERO_QKV_GROUP']='invalid'
        try:helper.zero_weight_low_linear(x[:2],parts,count,output_groups=3)
        except ValueError as e:assert 'group' in str(e)
        else:raise AssertionError('Invalidgroupaccepted')
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        os.environ['UNIFOLM_WORLD_ZERO_QKV_GROUP']='1-large'
    print('QKV_GROUP1_RUNTIME '+json.dumps(dict(passed=True,real_QKV_fixtures=4,policy_fixtures=8,
        old_master_cache_graph_fallback_and_stage_guards=True,small_M640_stays32=True,
        mode_backend_bias_outputgroup_tail_fallbacks=True,invalid_mode_rejected=True,
        actual_kernel_launches_observed=len(launches))),flush=True)
    print('QKV_GROUP1_RUNTIME_CHECK passed=true fixtures=4 policies=8',flush=True)


if __name__=='__main__':main()
