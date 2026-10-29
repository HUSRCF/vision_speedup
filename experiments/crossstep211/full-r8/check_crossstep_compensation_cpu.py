"""Check correlation statistics, retained snapshots and input immutability."""
import torch
from crossstep_compensation_observer import sample

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    values=torch.arange(40*96,dtype=torch.float32).reshape(40,96)
    high=(values.sin()*.7).half();low=(values.cos()*.2).half()
    wh=(torch.arange(80*96,dtype=torch.float32).reshape(80,96).cos()*.03).half();wl=torch.zeros_like(wh)
    saved=[x.clone()for x in (high,low,wh,wl)];rng=torch.get_rng_state().clone()
    first,previous=sample(high,low,wh,wl,None)
    same,_=sample(high,low,wh,wl,previous);assert same['reuse_relative_l2']==0 and abs(same['correction_cosine']-1)<1e-12
    opposite,_=sample(high,-low,wh,wl,previous)
    assert abs(opposite['correction_cosine']+1)<1e-12 and abs(opposite['reuse_vs_drop_error_ratio']-2)<1e-6
    assert first['correction_relative_l2']>0 and all(torch.equal(a,b)for a,b in zip(saved,(high,low,wh,wl)))
    low.zero_();assert torch.count_nonzero(previous[1])>0,'Previous low must be an owned snapshot'
    assert torch.equal(rng,torch.get_rng_state())and not torch.cuda.is_initialized()
    print('COMPENSATION_CORRELATION_CPU passed=true identical_reuse_error0=true opposite_error_ratio2=true snapshots_owned=true input_immutable=true RNG_unchanged=true no_GPU=true')

if __name__=='__main__':main()
