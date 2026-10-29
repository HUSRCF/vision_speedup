"""Exact 12-round cache fidelity and explicit stale-cache rejection cases."""
import copy,torch
from static_timestep_cache import Controller

class ResBlock(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb_layers=torch.nn.Sequential(torch.nn.SiLU(),torch.nn.Linear(32,16));self.projection=torch.nn.Linear(16,16)
    def forward(self,x,emb):return torch.tanh(self.projection(x)+self.emb_layers(emb))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.time_embed=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.SiLU(),torch.nn.Linear(32,32));self.fps_embedding=copy.deepcopy(self.time_embed);self.blocks=torch.nn.ModuleList([ResBlock()for _ in range(4)])
    def forward(self,x,t,fs):
        # Both stage and dynamic x vary; only the declared time/fs inputs feed
        # the selected MLPs, matching the source-verified WMA dependency.
        emb=self.time_embed(torch.arange(8,dtype=x.dtype).view(1,8)+t.to(x.dtype)/50)
        emb=emb.repeat_interleave(16,0)+self.fps_embedding(torch.full((1,8),float(fs),dtype=x.dtype)).repeat_interleave(16,0)
        for block in self.blocks:x=block(x,emb)
        return x


def selected(model):
    return dict(time_embed=model.time_embed,fps_embedding=model.fps_embedding,**{f'blocks.{i}.emb_layers':b.emb_layers for i,b in enumerate(model.blocks)})

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized();torch.manual_seed(7)
    for dtype in(torch.float32,torch.float16):
        original=Model().eval().to(dtype);model=copy.deepcopy(original);controller=Controller(selected(model));rng=torch.get_rng_state().clone()
        for generation in range(12):
            for stage in('decision','world'):
                for position in range(50):
                    t=torch.tensor([49-position]);fs=torch.tensor([4]);x=torch.full((16,16),(generation+.1*position+(stage=='world'))/100,dtype=dtype)
                    read=controller.begin(stage=stage,generation=generation,position=position,timesteps=t,fs=fs,backend_key=(stage,dtype));assert read==(generation>0)
                    actual=model(x,t,4);reference=original(x,t,4)
                    assert torch.equal(actual,reference),(dtype,generation,stage,position)
                    controller.end()
        r=controller.report();assert r['counts']==dict(decision_calls=600,decision_fill=50,world_calls=600,world_fill=50,decision_read=550,world_read=550)
        assert len(r['completed_generations'])==24 and all(g['steps']==50 for g in r['completed_generations'])
        assert torch.equal(rng,torch.get_rng_state());controller.close()
    for mutation in('time','fs','backend','weight','replacement','skip','unfinished_generation'):
        model=Model().eval();c=Controller(selected(model));x=torch.zeros(16,16)
        for position in range(50):
            c.begin(stage='world',generation=0,position=position,timesteps=torch.tensor([49-position]),fs=torch.tensor([4]),backend_key='fixed');model(x,torch.tensor([49-position]),4);c.end()
        t=torch.tensor([49]);fs=torch.tensor([4]);backend='fixed';pos=0
        if mutation=='time':t=torch.tensor([48])
        if mutation=='fs':fs=torch.tensor([5])
        if mutation=='backend':backend='changed'
        if mutation=='weight':model.time_embed[0].weight.add_(.1)
        if mutation=='replacement':model.time_embed[0].weight=torch.nn.Parameter(model.time_embed[0].weight.detach().clone())
        if mutation=='skip':pos=1
        if mutation=='unfinished_generation':
            c.begin(stage='world',generation=1,position=0,timesteps=t,fs=fs,backend_key=backend);model(x,t,4);c.end()
        try:c.begin(stage='world',generation=2 if mutation=='unfinished_generation'else 1,position=pos,timesteps=t,fs=fs,backend_key=backend)
        except AssertionError:pass
        else:raise AssertionError('Stale/invalid cache use accepted: '+mutation)
        c.close()
    assert not torch.cuda.is_initialized()
    print('STATIC_TIMESTEP_CPU passed=true FP32_and_Half_2400_full_outputs_bitwise=true 24x50_each_dtype=true 7_rejection_cases=true RNG_unchanged=true no_GPU=true no_dynamic_feature_cache=true')

if __name__=='__main__':main()
