"""Exercise actual WMA ResBlock and static cache through the graph executor."""
import copy
from types import MethodType,SimpleNamespace
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.networks.wma_model import ResBlock
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from unifolm_wma.utils.static_timestep_runtime import Runtime

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fs_condition=True;self.default_fs=4
        self.time_embed=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.SiLU(),torch.nn.Linear(32,32))
        self.fps_embedding=copy.deepcopy(self.time_embed)
        self.block=ResBlock(32,32,0)
        # The original block's output convolution starts at zero. Give this
        # CPU toy a nonzero path so stale embedding errors affect its output.
        with torch.no_grad():self.block.out_layers[-1].weight.normal_(0,.01)
        self.live_convolution_calls=0
    def forward(self,x,action,state,t,condition,*,fs,**kwargs):
        time=torch.arange(8,dtype=x.dtype).reshape(1,8)+t.to(x.dtype).reshape(1,1)/50
        fps=torch.arange(8,dtype=x.dtype).reshape(1,8)+fs.to(x.dtype).reshape(1,1)
        emb=self.time_embed(time).repeat_interleave(16,0)+self.fps_embedding(fps).repeat_interleave(16,0)
        return self.block(x,emb,batch_size=1)

class Executor:
    def __init__(self,model):
        self.denoiser_graph=DenoiserGraph(model.forward);self.calls=0
    def __call__(self,*args,**kwargs):
        assert '_static_timestep_read'in kwargs;self.calls+=1
        return self.denoiser_graph(*args,**kwargs)

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized();torch.manual_seed(11)
    original=Model().eval();model=copy.deepcopy(original);executor=Executor(model)
    def count(*unused):model.live_convolution_calls+=1
    model.block.in_layers[-1].register_forward_hook(count)
    outer=SimpleNamespace(model=SimpleNamespace(diffusion_model=model),apply_model=executor)
    runtime=Runtime(outer);assert len(runtime.controller.modules)==3
    assert runtime.executor is executor and runtime.graph is executor.denoiser_graph and runtime.graph.max_graphs==4
    rng=torch.get_rng_state().clone()
    for generation in range(2):
        for stage in('decision','world'):
            for position in range(50):
                t=torch.tensor([49-position]);fs=torch.tensor([4]);x=torch.full((16,32,2,2),(generation+.01*position+(stage=='world'))/10)
                args=(x,None,None,t,{'dynamic_condition':generation+position});runtime.set_step(stage,position)
                actual=outer.apply_model(*args,fs=fs);reference=original(*args,fs=fs)
                assert torch.equal(actual,reference),(generation,stage,position)
                runtime.finish_step()
    r=runtime.report();assert r['counts']==dict(decision_calls=100,decision_fill=50,world_calls=100,world_fill=50,decision_read=50,world_read=50)
    assert executor.calls==200 and model.live_convolution_calls==200
    assert torch.equal(rng,torch.get_rng_state())and not torch.cuda.is_initialized()
    runtime.controller.close()
    print('STATIC_TIMESTEP_RUNTIME_CPU passed=true actual_WMA_ResBlock=true all200_outputs_bitwise=true all200_parallel_executor_calls=true dynamic_convolution_live=true no_GPU=true')

if __name__=='__main__':main()
