"""Original real-block/cache/fallback guards plus actual launch policy."""
import json
import os
import torch
from check_world_geglu_hidden_parts import main as original_guards
from unifolm_wma.utils import geglu_hidden_parts_pipeline as pipeline
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_zero_weight_low_geglu import residual_count


@torch.no_grad()
def main():
    assert os.environ['UNIFOLM_WORLD_GEGLU_C640_GROUP']=='4'
    original=pipeline._zero_geglu_hidden_parts;launches=[]
    class Observed:
        def __getitem__(self,grid):
            launch=original[grid]
            def call(*args,**kwargs):
                launches.append((args[10],args[11],args[12],args[17]))
                assert kwargs['num_warps']==4 and kwargs['num_stages']==2
                return launch(*args,**kwargs)
            return call
    pipeline._zero_geglu_hidden_parts=Observed()
    try:
        original_guards()
        assert launches and all(r==(10240,2560,640,4) for r in launches)
        eligible=len(launches)
        x=pack_half_weight(torch.randn(13,640,device='cuda')*.1,weight_scale=1.)
        w=pack_half_weight(torch.randn(5120,640,device='cuda')*.01)
        count=residual_count(w,refresh=True)
        actual=pipeline.geglu_hidden_parts(x,w,count)
        assert launches[-1]==(13,2560,640,32)
        os.environ['UNIFOLM_WORLD_GEGLU_C640_GROUP']='32'
        reference=pipeline.geglu_hidden_parts(x,w,count)
        assert launches[-1]==(13,2560,640,32)
        assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(actual,reference))
        os.environ['UNIFOLM_WORLD_GEGLU_C640_GROUP']='invalid'
        try:pipeline.geglu_hidden_parts(x,w,count)
        except ValueError as exc:assert str(exc)=='Invalid world C640 GEGLU group'
        else:raise AssertionError('invalid group accepted')
    finally:
        os.environ['UNIFOLM_WORLD_GEGLU_C640_GROUP']='4'
        pipeline._zero_geglu_hidden_parts=original
    print('GEGLU640_GROUP4_RUNTIME_CHECK '+json.dumps(dict(passed=True,real_block_fixtures=2,eligible_launches=eligible,
        actual_world_shape_group4=True,other_shapes_group32=True,explicit_group32=True,invalid_flag_rejected=True)),flush=True)


if __name__=='__main__':main()
