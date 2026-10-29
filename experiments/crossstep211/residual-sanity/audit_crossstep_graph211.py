"""Small CUDA-graph correctness check before allocating the real WMA model."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from guard_crossstep211 import announce_worker
from check_crossstep_feature_reuse_cpu import Model
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from unifolm_wma.utils.crossstep_feature_runtime import Runtime

class CheckedGraph(DenoiserGraph):
    def __call__(self,*args,**kwargs):
        expected=self.original(*args,**kwargs)
        out=super().__call__(*args,**kwargs)
        assert all(torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))for a,b in zip(out,expected))
        return out

class Owner(torch.nn.Module):
    def __init__(self):
        super().__init__();self.diffusion=Model();self.model=SimpleNamespace(diffusion_model=self.diffusion)
    def apply_model(self,x,a,s,t,cond,**kwargs):
        h=self.diffusion(x+t[:,None].float()*.001+cond['value']*.01)
        return h+a*.01,h+s*.01,h

@torch.no_grad()
def main():
    hold=announce_worker();torch.manual_seed(21150);owner=Owner().cuda().eval();owner.apply_model=CheckedGraph(owner.apply_model);r=Runtime(owner,input_start=1,output_end=1,boundary_residual=True)
    rng=torch.cuda.get_rng_state().clone();changed=[]
    for generation in range(2):
        for stage in ('decision','world'):
            torch.backends.cudnn.allow_tf32=stage=='decision';cond={'value':torch.full((1,4),.1+generation,device='cuda')}
            for position in range(50):
                if stage=='world' and generation==0 and position==23:owner.diffusion.middle_block.weight.add_(.01);changed.append('master')
                if stage=='world' and generation==0 and position==27:cond['value'].add_(.02);changed.append('condition')
                x=torch.full((1,4),position*.01,device='cuda');t=torch.tensor([981-position*20],device='cuda',dtype=torch.long)
                r.set_step(stage,position);out=owner.apply_model(x,x,x,t,cond);r.finish_step();assert len(out)==3 and all(torch.isfinite(v).all()for v in out)
    torch.cuda.synchronize();assert torch.equal(rng,torch.cuda.get_rng_state());report=r.report();assert report['graph_count']==3 and report['counts']['decision_calls']==report['counts']['world_calls']==100 and report['invalidations']['master_version']==1 and report['invalidations']['condition_change']==1
    for entry in r.graph.entries.values():entry['graph'].reset()
    r.graph.entries.clear();r.controller.close()
    report.update(diagnostic_complete=True,all_dynamic_gates_passed=True,synthetic_only=True,real_WMA_or_video_quality_verified=False,three_outputs_checked_calls=200,RNG_unchanged=True,changed=changed)
    Path('comparison.json').write_text(json.dumps(report,indent=2)+'\n');print('CROSSSTEP_GRAPH211 complete=true checks=600',flush=True)

if __name__=='__main__':main()
