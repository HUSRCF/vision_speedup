"""Preserve both-stage shared master/cache guards and prove scoped dispatch."""
import json
import os
import torch
from check_ffn1280_zero_stages import main as previous_guard
from geglu_c1280_subtile32_hooks import installed
from unifolm_wma.utils.geglu_hidden_parts_c1280_pipeline import geglu_hidden_parts_c1280
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight


def exact(a,b):
    for x,y in zip(a,b):
        assert x.shape==y.shape and x.dtype==y.dtype==torch.float16
        assert torch.equal(x.contiguous().view(torch.int16),y.contiguous().view(torch.int16))


@torch.no_grad()
def main():
    os.environ['UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32']='1'
    os.environ['UNIFOLM_WORLD_GEGLU_C1280_STAGES']='3'
    os.environ['UNIFOLM_WORLD_FFN1280_ZERO_STAGES']='3'
    with installed() as launches:
        previous_guard()
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        weight=torch.randn(10240,1280,device='cuda')*.02
        parts=pack_half_weight(weight);count=torch.count_nonzero(parts[1])
        bias=torch.randn(10240,device='cuda')*.02
        for rows in [2560,640,13]:
            value=pack_half_weight(torch.randn(rows,1280,device='cuda')*.2,weight_scale=1.)
            for policy in ['selected','disabled','decision','matmul_tf32']:
                torch.backends.cudnn.allow_tf32=policy=='decision'
                torch.backends.cuda.matmul.allow_tf32=policy=='matmul_tf32'
                os.environ['UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32']='0'
                expected=geglu_hidden_parts_c1280(value,parts,count,bias)
                os.environ['UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32']='0' if policy=='disabled' else '1'
                before=len(launches);actual=geglu_hidden_parts_c1280(value,parts,count,bias)
                exact(actual,expected);assert len(launches)==before+1
                assert launches[-1]['subtile32']==(policy=='selected' and rows in {2560,640})
            del value,expected,actual
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        os.environ['UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32']='invalid'
        value=pack_half_weight(torch.randn(13,1280,device='cuda'),weight_scale=1.)
        try:geglu_hidden_parts_c1280(value,parts,count,bias)
        except ValueError as e:assert 'subtile' in str(e)
        else:raise AssertionError('Invalidsubtileaccepted')
        os.environ['UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32']='1'
    print('GEGLU_C1280_SUBTILE32_RUNTIME '+json.dumps(dict(passed=True,real_block_fixtures=8,policy_fixtures=12,
        original_shared_master_version_cache_and_both_stage_guards=True,production_host_dispatch=True,
        selected_2_shapes_and_backend_mode_tail_fallbacks=True,invalid_mode_rejected=True,
        actual_kernel_launches_observed=len(launches))),flush=True)
    print('GEGLU_C1280_SUBTILE32_RUNTIME_CHECK passed=true fixtures=8 policies=12',flush=True)


if __name__=='__main__':main()
