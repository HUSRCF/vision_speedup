"""Two-refresh predictor: affine fidelity, real step gaps and stale rejection."""
import copy
import torch
from unifolm_wma.utils.crossstep_feature_reuse import Controller,predict

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.5));self.position=0;self.calls=0
    def forward(self,x):
        self.calls+=1
        return x*self.weight+.125+self.position/64

class SpatialTransformer(Block):pass

class Model(torch.nn.Module):
    def __init__(self,spatial=False):
        super().__init__();kind=SpatialTransformer if spatial else Block
        self.input_blocks=torch.nn.ModuleList([kind()for _ in range(3)]);self.middle_block=kind();self.output_blocks=torch.nn.ModuleList([kind()for _ in range(3)]);self.head=Block();self.position=0
    def forward(self,x):
        for module in [*self.input_blocks,self.middle_block,*self.output_blocks,self.head]:module.position=self.position;x=module(x)
        return x

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    for scope,boundary in(('deep',False),('deep',True),('spatial',False)):
        original=Model(scope=='spatial').eval();model=copy.deepcopy(original)
        c=Controller(model,input_start=1,output_end=1,scope=scope,boundary_residual=boundary,stages=('decision','world'))
        rng=torch.get_rng_state().clone()
        for generation in range(12):
            for stage in('decision','world'):
                for position in range(50):
                    model.position=original.position=position
                    x=torch.full((1,17),generation/8+(stage=='world')/8+position/64)
                    reuse=c.begin(stage=stage,generation=generation,position=position,total_steps=50,condition_key=(stage,generation),input_key=(1,17))
                    assert reuse==(10<=position<40 and position%2==1)
                    actual=model(x);expected=original(x)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    if reuse:assert float(c.alpha)==(1. if position==11 else .5)
                    c.end()
        r=c.report();assert r['alpha_histogram']=={'1.0':24,'0.5':336}
        assert r['counts']==dict(decision_calls=600,decision_refresh=420,decision_reuse=180,world_calls=600,world_refresh=420,world_reuse=180)
        assert len(r['completed_generations'])==24 and model.head.calls==1200
        if scope=='deep':assert model.input_blocks[0].calls==model.output_blocks[-1].calls==1200
        assert r['predictive_feature_reuse']and r['residual_cache_FP32']and torch.equal(rng,torch.get_rng_state())
        c.close()
    for changed in('master','condition'):
        model=Model().eval();c=Controller(model,input_start=1,output_end=1)
        for position in range(17):
            model.position=position
            if position==12 and changed=='master':model.middle_block.weight.add_(.125)
            condition='new'if changed=='condition'and position>=12 else'old'
            use=c.begin(stage='world',generation=0,position=position,total_steps=50,condition_key=condition,input_key=(1,17))
            if position in(12,13):assert not use,'Need two fresh full evaluations after invalidation'
            if position==15:assert use and float(c.alpha)==1.
            model(torch.full((1,17),position/64));c.end()
        c.close()
    # Repeated warm-up at a single position must not count as two past steps.
    model=Model().eval();c=Controller(model,input_start=1,output_end=1)
    for position in range(50):
        model.position=position
        c.begin(stage='world',generation=0,position=position,total_steps=50,condition_key=0,input_key=(1,17))
        value=model(torch.full((1,17),position/64))
        if position==0:
            for _ in range(4):model(torch.full((1,17),position/64))
        c.end()
    assert c.report()['counts']['world_reuse']==15;c.close()
    for dtype in(torch.float16,torch.float32):
        for residual in(False,True):
            x=torch.arange(257,dtype=torch.float32)/256;previous=x-.125;alpha=torch.tensor([.5]);live=x.to(dtype)if residual else None
            expected=x+(x-previous)*alpha
            if residual:expected=expected+live.float()
            assert torch.equal(predict(x,previous,alpha,live,output_dtype=dtype),expected.to(dtype))
    # A generation may not skip any sampler position.
    model=Model().eval();c=Controller(model,input_start=1,output_end=1)
    c.begin(stage='world',generation=0,position=0,total_steps=50,condition_key=0,input_key=(1,17));model(torch.zeros(1,17));c.end()
    try:c.begin(stage='world',generation=0,position=2,total_steps=50,condition_key=0,input_key=(1,17))
    except AssertionError:pass
    else:raise AssertionError('Skipped position accepted')
    c.close();assert not torch.cuda.is_initialized()
    print('CROSSSTEP_PREDICT_CPU passed=true affine_complete_outputs=3600 alpha_first1_steadyHalf=true two_fresh_after_invalidation=true warmup_not_extra_step=true FP32_residual=true RNG_unchanged=true no_GPU=true')

if __name__=='__main__':main()

from unifolm_wma.utils.crossstep_feature_runtime import sampler_stage
for math_flag in (False,True):
    torch.backends.cudnn.allow_tf32=math_flag
    for sim_mode in (False,True):
        assert sampler_stage((None,None,None,{'c_crossattn_action':[None,None,sim_mode,False]}),{})==('world'if sim_mode else'decision')
assert not torch.cuda.is_initialized()
print('NONV2_EXPLICIT_STAGE_CPU passed=true independent_of_TF32=true')
