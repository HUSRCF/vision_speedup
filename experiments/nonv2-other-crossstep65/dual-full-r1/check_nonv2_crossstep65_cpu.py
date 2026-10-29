"""Verify module contracts and that reuse calls retain the parallel executor."""
import json,torch
from pathlib import Path
from types import MethodType,SimpleNamespace
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from unifolm_wma.utils.crossstep_feature_runtime import Runtime
from unifolm_wma.utils.nccl_nonv2_all_blocks_runtime63 import expected_modules

class Block(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.8));self.calls=0
    def forward(self,x):self.calls+=1;return torch.tanh(x*self.weight+.01)

class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__();self.input_blocks=torch.nn.ModuleList([Block()for _ in range(3)]);self.middle_block=Block();self.output_blocks=torch.nn.ModuleList([Block()for _ in range(3)]);self.head=Block()
    def forward(self,x):
        for block in self.input_blocks:x=block(x)
        x=self.middle_block(x)
        for block in self.output_blocks:x=block(x)
        return self.head(x)

class Executor:
    def __init__(self,model):
        def original(owner,*args,**kwargs):return owner(args[0])
        self.denoiser_graph=DenoiserGraph(MethodType(original,model));self.calls=0
    def __call__(self,*args,**kwargs):
        self.calls+=1;assert '_crossstep_feature_reuse'in kwargs
        return self.denoiser_graph(*args,**kwargs)

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    proof=json.loads((Path(__file__).parent/'module-coverage.json').read_text())
    names=tuple(sorted(proof['full_module_names']));assert len(names)==33
    for row in proof['candidates'].values():
        result=expected_modules(names,tuple(row['selected']),row['scope'],True)
        assert result==tuple(sorted(row['retained']))and len(result)==row['count']
        assert expected_modules(names,tuple(row['selected']),row['scope'],False)==names
    try:expected_modules(names[:-1]+(names[0],),(),'deep',False)
    except AssertionError:pass
    else:raise AssertionError('Duplicate module name accepted')
    backbone=Backbone().eval();executor=Executor(backbone)
    outer=SimpleNamespace(model=SimpleNamespace(diffusion_model=backbone),apply_model=executor)
    runtime=Runtime(outer,input_start=1,output_end=1,stages=('decision','world'))
    assert runtime.executor is executor and runtime.graph is executor.denoiser_graph and runtime.graph.max_graphs==4
    rng=torch.get_rng_state().clone();x=torch.ones(1,4)
    for generation in range(12):
        for stage in ('decision','world'):
            condition={'stage':stage,'generation':generation}
            for position in range(50):
                runtime.set_step(stage,position)
                outer.apply_model(x+position*.001,None,None,torch.tensor([49-position]),condition,fs=None)
                runtime.finish_step()
    report=runtime.report()
    assert executor.calls==1200
    assert report['counts']==dict(decision_calls=600,decision_refresh=420,decision_reuse=180,world_calls=600,world_refresh=420,world_reuse=180)
    assert len(report['completed_generations'])==24 and all(g['steps']==50 for g in report['completed_generations'])
    assert backbone.input_blocks[0].calls==backbone.output_blocks[-1].calls==backbone.head.calls==1200
    assert backbone.middle_block.calls==840
    assert torch.equal(rng,torch.get_rng_state())and not torch.cuda.is_initialized()
    runtime.controller.close()
    print('NONV2_CROSSSTEP65_CPU passed=true all1200_calls_through_parallel_executor=true role_module_sets=true duplicate_rejected=true no_GPU=true full_video_quality_not_inferred=true')

if __name__=='__main__':main()
