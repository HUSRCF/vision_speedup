"""Check sampler fidelity, fresh shallow/head work and cache invalidation."""
import torch
from unifolm_wma.utils.crossstep_feature_reuse import Controller

class Block(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.8));self.calls=0
    def forward(self,x):self.calls+=1;return torch.tanh(x*self.weight+.01)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.input_blocks=torch.nn.ModuleList([Block()for _ in range(3)]);self.middle_block=Block();self.output_blocks=torch.nn.ModuleList([Block()for _ in range(3)]);self.head=Block()
    def forward(self,x):
        for block in self.input_blocks:x=block(x)
        x=self.middle_block(x)
        for block in self.output_blocks:x=block(x)
        return self.head(x)

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized();model=Model().eval();c=Controller(model,input_start=1,output_end=1);x=torch.ones(1,4);rng=torch.get_rng_state().clone();reuse=0
    for generation in range(11):
        for stage in ('decision','world'):
            for position in range(50):
                use=c.begin(stage=stage,generation=generation,position=position,total_steps=50,condition_key=(stage,generation),input_key=tuple(x.shape));reuse+=use
                model(x+position*.001);c.end()
    r=c.report();assert r['counts']=={'decision_calls':550,'decision_refresh':550,'world_calls':550,'world_refresh':385,'world_reuse':165}
    assert len(r['completed_generations'])==22 and reuse==165
    assert model.input_blocks[0].calls==model.output_blocks[-1].calls==model.head.calls==1100
    assert model.input_blocks[1].calls==model.middle_block.calls==model.output_blocks[0].calls==935
    assert torch.equal(rng,torch.get_rng_state())
    c.close()
    # Changes to weights and conditioning force a full refresh at a nominal hit.
    for changed in ('master','condition'):
        model=Model().eval();c=Controller(model,input_start=1,output_end=1)
        for position in range(12):
            if position==11 and changed=='master':model.middle_block.weight.add_(.1)
            key='changed'if position==11 and changed=='condition'else'original'
            use=c.begin(stage='world',generation=0,position=position,total_steps=50,condition_key=key,input_key=tuple(x.shape));assert not use
            model(x);c.end()
        assert c.last_invalidation;c.close()
    # No stale feature from the prior generation can be read at a new start.
    model=Model().eval();c=Controller(model,input_start=1,output_end=1)
    for position in range(50):
        c.begin(stage='world',generation=0,position=position,total_steps=50,condition_key=0,input_key=tuple(x.shape));model(x);c.end()
    assert not c.begin(stage='world',generation=1,position=0,total_steps=50,condition_key=0,input_key=tuple(x.shape));model(x);c.end()
    try:c.begin(stage='world',generation=1,position=2,total_steps=50,condition_key=0,input_key=tuple(x.shape))
    except AssertionError:pass
    else:raise AssertionError('Skipped sampler position accepted')
    c.close();assert not torch.cuda.is_initialized()
    print('CROSSSTEP_FEATURE_CPU passed=true full11_rounds=true full50_steps=true calls=1100 shallow_and_heads=1100 selected_refresh=935 selected_reuse=165 master_condition_and_generation_invalidation=true no_GPU=true')

if __name__=='__main__':main()

from unifolm_wma.utils.crossstep_feature_runtime import sampler_stage
for math_flag in (False,True):
    torch.backends.cudnn.allow_tf32=math_flag
    for sim_mode in (False,True):
        assert sampler_stage((None,None,None,{'c_crossattn_action':[None,None,sim_mode,False]}),{})==('world'if sim_mode else'decision')
assert not torch.cuda.is_initialized()
print('NONV2_EXPLICIT_STAGE_CPU passed=true independent_of_TF32=true')
