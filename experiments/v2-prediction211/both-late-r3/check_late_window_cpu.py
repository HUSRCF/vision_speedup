"""Check actual late-window selection on an affine 12-input/12-output model."""
import copy
import torch
import check_crossstep_feature_reuse_cpu as toy
from unifolm_wma.utils.crossstep_feature_reuse import Controller


@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    model=toy.Model().eval()
    model.input_blocks=torch.nn.ModuleList([toy.Block()for _ in range(12)])
    model.output_blocks=torch.nn.ModuleList([toy.Block()for _ in range(12)])
    for parameter in model.parameters():parameter.fill_(1)
    original=copy.deepcopy(model)
    controller=Controller(model,input_start=6,output_end=6,start=20,stop=40,
                          boundary_residual=False,scope='deep',stages=('decision','world'))
    rng=torch.get_rng_state().clone()
    for generation in range(11):
        for stage in('decision','world'):
            for position in range(50):
                original.position=model.position=position
                live=torch.full((1,17),generation/8+(stage=='world')/8+position/64)
                reuse=controller.begin(stage=stage,generation=generation,position=position,
                                       total_steps=50,condition_key=(stage,generation),input_key=(1,17))
                assert reuse==(20<=position<40 and position%2==1)
                torch.testing.assert_close(model(live),original(live),rtol=0,atol=0)
                controller.end()
    report=controller.report()
    assert len(report['selected_blocks'])==13
    assert report['alpha_histogram']=={'1.0':22,'0.5':198}
    assert report['counts']==dict(decision_calls=550,decision_refresh=440,decision_reuse=110,
                                 world_calls=550,world_refresh=440,world_reuse=110)
    assert len(report['completed_generations'])==22 and model.head.calls==1100
    assert model.input_blocks[0].calls==model.output_blocks[-1].calls==1100
    assert not report['sampler_step_count_reduced'] and not report['round_count_reduced']
    assert torch.equal(rng,torch.get_rng_state())
    controller.close()
    assert not torch.cuda.is_initialized()
    print('V2_LATE_WINDOW_CPU passed=true calls1100=true selected13=true reuse220=true affine_exact=true no_GPU=true')


if __name__=='__main__':main()
