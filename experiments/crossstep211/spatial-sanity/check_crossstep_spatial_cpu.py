"""Check per-module residual reuse while every convolution/head stays live."""
import torch
from crossstep_feature_reuse import Controller
from check_crossstep_feature_reuse_cpu import Block,Model

class SpatialTransformer(Block):
    def forward(self,x):return x+super().forward(x)

def spatial_model():
    model=Model();model.input_blocks[1]=SpatialTransformer();model.middle_block=SpatialTransformer()
    return model

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized();model=spatial_model().eval()
    c=Controller(model,input_start=1,output_end=1,scope='spatial');assert len(c.saved)==2
    handles=[];checks=[]
    for module,_,name in c.saved:
        def check(m,args,out,name=name):
            if c.current['stage']=='world'and c.current['reuse']:
                assert torch.equal(out,args[0]+c.buffers[('world',name)]);checks.append(name)
        handles.append(module.register_forward_hook(check))
    rng=torch.get_rng_state().clone()
    for generation in range(11):
        for stage in ('decision','world'):
            for position in range(50):
                c.begin(stage=stage,generation=generation,position=position,total_steps=50,condition_key=(stage,generation),input_key=(1,4))
                model(torch.full((1,4),.2+position*.03));c.end()
    r=c.report();assert r['counts']['decision_calls']==r['counts']['world_calls']==550 and r['counts']['world_reuse']==165
    assert model.input_blocks[0].calls==model.input_blocks[2].calls==model.output_blocks[0].calls==model.head.calls==1100
    assert model.input_blocks[1].calls==model.middle_block.calls==935 and len(checks)==330
    assert torch.equal(rng,torch.get_rng_state())and r['all_convolutions_and_heads_live']and r['per_module_residual']
    for h in handles:h.remove()
    c.close();print('CROSSSTEP_SPATIAL_CPU passed=true calls1100=true conv_and_heads1100=true selected935=true current_input_plus_delta_checks330=true no_GPU=true')

if __name__=='__main__':main()
