"""Check fresh boundary propagation and refresh invalidation of residual reuse."""
import torch
from crossstep_feature_reuse import Controller
from check_crossstep_feature_reuse_cpu import Model

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    model=Model().eval();c=Controller(model,input_start=1,output_end=1,boundary_residual=True)
    entry=[];exit=[]
    a=model.input_blocks[1].register_forward_pre_hook(lambda m,args:entry.append(args[0].clone()))
    b=model.output_blocks[0].register_forward_hook(lambda m,args,out:exit.append(out.clone()))
    rng=torch.get_rng_state().clone();delta=None
    for position in range(50):
        use=c.begin(stage='world',generation=0,position=position,total_steps=50,condition_key=0,input_key=(1,4))
        model(torch.full((1,4),.2+position*.03))
        if use:assert torch.equal(exit[-1],entry[-1]+delta)
        else:delta=(exit[-1]-entry[-1]).clone()
        c.end()
    report=c.report();assert report['counts']['world_calls']==50 and report['counts']['world_reuse']==15
    assert report['boundary_residual']and report['current_boundary_input_live']
    assert len(entry)==len(exit)==50 and model.middle_block.calls==35
    assert not torch.equal(entry[10],entry[11]) and torch.equal(rng,torch.get_rng_state())
    a.remove();b.remove();c.close()
    print('CROSSSTEP_RESIDUAL_CPU passed=true fresh_boundary_input=true live50_steps=true reuse15=true RNG_unchanged=true no_GPU=true')

if __name__=='__main__':main()
