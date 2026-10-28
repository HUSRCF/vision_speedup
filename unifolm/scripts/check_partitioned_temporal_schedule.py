"""Retain all real cache/native guards, then observe scoped scheduling policy."""
import json
import os
import torch
from check_world_partitioned_temporal import main as previous_guard
from partitioned_temporal_schedule_hooks import installed
from unifolm_wma.utils.partitioned_temporal import partitioned_temporal_conv
from unifolm_wma.utils.triton_compensated_temporal_conv import temporal_weight_parts


def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
    assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    os.environ['UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE']='3'
    with installed() as launches:
        previous_guard()
        for channels,h,w in [(320,40,64),(640,20,32)]:
            value=torch.randn(1,channels,16,h,w,device='cuda')*.2
            weight=torch.randn(channels,channels,3,1,1,device='cuda')*.02
            parts=temporal_weight_parts(weight);count=torch.count_nonzero(parts[1])
            group=8 if channels==320 else 16
            for policy in ['selected','disabled','decision','matmul_tf32','tail']:
                torch.backends.cudnn.allow_tf32=policy=='decision'
                torch.backends.cuda.matmul.allow_tf32=policy=='matmul_tf32'
                x=value[:,:,:3].contiguous() if policy=='tail' else value
                os.environ['UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE']='2'
                reference=partitioned_temporal_conv(x,parts,count,group_m=group)
                os.environ['UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE']='2' if policy=='disabled' else '3'
                before=len(launches);actual=partitioned_temporal_conv(x,parts,count,group_m=group)
                exact(actual,reference);assert len(launches)==before+2
                two,three=launches[-2:]
                assert not two['three'] and three['three'] and three['group']==group and three['stages']==2
                assert two['stages']==(3 if policy=='selected' else 2)
                assert two['group']==(32 if policy=='selected' and channels==320 else group)
            os.environ['UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE']='invalid'
            try:partitioned_temporal_conv(value,parts,count,group_m=group)
            except ValueError as e:assert 'schedule' in str(e)
            else:raise AssertionError('Invalidscheduleaccepted')
            torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            os.environ['UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE']='3'
            del value,weight,parts,count,reference,actual,x
            torch.cuda.empty_cache()
    print('PARTITIONED_TEMPORAL_SCHEDULE_RUNTIME '+json.dumps(dict(passed=True,real_module_fixtures=4,
        scoped_policy_fixtures=10,all_original_cache_and_fallback_guards=True,original_three_product_schedule=True,
        mode_backend_tail_fallbacks=True,invalid_mode_rejected=True,actual_kernel_launches_observed=len(launches))),flush=True)
    print('PARTITIONED_TEMPORAL_SCHEDULE_RUNTIME_CHECK passed=true fixtures=4 policies=10',flush=True)


if __name__=='__main__':main()
