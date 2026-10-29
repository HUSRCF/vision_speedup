"""Synthetic six-state exact transport and parameter/buffer alias checks."""
import json,os
from pathlib import Path
import torch
import torch.distributed as dist
from nccl_root_model_move import collect,move_model


class Fixture(torch.nn.Module):
    def __init__(self,noncontiguous=False):
        super().__init__()
        self.p=torch.nn.Parameter(torch.arange(4096,dtype=torch.float32).reshape(64,64)*.003)
        self.alias=self.p
        self.half_weight=torch.nn.Parameter(torch.arange(1024,dtype=torch.float16)*.002,requires_grad=False)
        self.empty=torch.nn.Parameter(torch.empty(0))
        value=torch.arange(256,dtype=torch.float32).reshape(16,16)
        self.register_buffer('values',value.T if noncontiguous else value)
        self.register_buffer('values_alias',self.values)
        self.register_buffer('ints',torch.arange(33,dtype=torch.int64))
        self.register_buffer('bools',torch.arange(17)%2==0)
        self.register_buffer('scalar',torch.tensor(-0.))
        self.register_buffer('zero',torch.empty(0,dtype=torch.float16))
        self.tracked_device=torch.device('cpu')

    def _apply(self,fn):
        result=super()._apply(fn)
        self.tracked_device=self.p.device
        return result


def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def main():
    cases=[];rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    for state in ('initial','changed','zero','restored','small_buckets','noncontiguous_fallback'):
        model=Fixture(state=='noncontiguous_fallback');original=model.p
        if state=='changed':
            for p in model.parameters():p.data.add_(.03125)
        if state=='zero':
            for p in model.parameters():p.data.zero_()
        if state=='restored':
            for p in model.parameters():
                saved=p.detach().clone();p.data.add_(.125);p.data.copy_(saved)
        expected={name:t.detach().clone()for name,t in model.state_dict().items()}
        if dist.get_rank()==1 and state!='noncontiguous_fallback':
            # Distinct worker contents prove values come from rank0, rather
            # than accidentally passing by independently moving equal models.
            with torch.no_grad():
                for e in collect(model):
                    t=e['tensor']
                    if t.dtype==torch.bool:t.logical_not_()
                    else:t.add_(3)
        model,report=move_model(model,bucket_bytes=32 if state=='small_buckets'else 4096)
        assert model.tracked_device.type=='cuda','Standard _apply device lifecycle must run'
        errors=[]
        for name,tensor in model.state_dict().items():
            same=bits(tensor.cpu(),expected[name]);assert same,(state,name)
            errors.append(dict(signedzero_bits=True))
        if state!='noncontiguous_fallback':assert model.p is original and model.alias is model.p and model.values is model.values_alias
        assert report['fallback_native_cuda']==(state=='noncontiguous_fallback')
        cases.append(dict(case=state,errors=errors,report=report))
    assert torch.equal(rng,torch.get_rng_state())and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    rank=dist.get_rank();out=Path(os.environ['NCCL_MODEL_SHARED'])/('rank'+str(rank)+'-report.json')
    report=dict(synthetic_only=True,no_official_score=True,all_supported_cases_bitwise=True,RNG_unchanged=True,cases=cases,pairs=[],no_kernel_argument_or_address_exports=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('ROOT_MODEL_MOVE_SYNTHETIC passed=true rank='+str(rank)+' cases=6 alias_and_dtype=true no_operand_export=true',flush=True)


if __name__=='__main__':main()
