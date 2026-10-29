"""Synthetic accuracy and complete two-step cost of FP16 incremental GEMM."""
import json,statistics
from pathlib import Path
import torch,triton
from guard_delta_gemm import announce_worker
from unifolm_wma.utils.ffn1280_zero_residual import _zero_low_gemm_residual
import temporal_delta_gemm_candidate as candidate

def reference(inputs,parts,count,bias,residual,output,bm,stages):
    m,k=inputs[0].shape;n=parts[0].shape[0]
    _zero_low_gemm_residual[(triton.cdiv(m,bm)*triton.cdiv(n,64),)](*inputs,*parts,count,parts[0]if bias is None else bias,residual,output,m,n,k,bias is not None,bm,32,1,num_warps=4,num_stages=stages,enable_fp_fusion=False)

def compare(a,b,exact=False):
    assert torch.isfinite(a).all()and torch.isfinite(b).all()
    difference=(a-b).double();relative=float(difference.norm()/b.double().norm().clamp_min(1e-30));maximum=float(difference.abs().max())
    bits=torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
    assert bits if exact else(relative<.001 and maximum<.001*max(1.,float(b.abs().max())))
    return dict(relative_l2=relative,max_abs=maximum,signedzero_bits=bits,exact_required=exact,finite=True)

def elapsed(graph,repeats=30):
    first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record()
    for _ in range(repeats):graph.replay()
    last.record();last.synchronize();return first.elapsed_time(last)/repeats

@torch.no_grad()
def screen(m,n,k,bm,stages,three):
    x0=(torch.randn((m,k),device='cuda')*.3).half();l0=(torch.randn((m,k),device='cuda')*.1).half()
    noise=torch.randn_like(x0)*.3
    xs=[(x0,l0)]+[((x0.float()+noise.float()*alpha).half(),(torch.randn_like(l0)*.1).half())for alpha in (.01,.05,.2)]
    wh=(torch.randn((n,k),device='cuda')*.03).half();wl=(torch.randn_like(wh)*.03).half()if three else torch.zeros_like(wh)
    parts=(wh,wl);count=torch.tensor(int(three),device='cuda',dtype=torch.int64)
    bias=(torch.randn(n,device='cuda')*.01);residual=torch.randn((m,n),device='cuda')*.05
    saved_wh=wh.clone();saved_wl=wl.clone();saved_bias=bias.clone();saved_residual=residual.clone()
    out=torch.empty_like(residual);expected=torch.empty_like(residual);cache=candidate.allocate(x0,n)
    rng=torch.cuda.get_rng_state().clone();states=[]
    candidate.run(xs[0],parts,count,bias,residual,out,cache,refresh=True,bm=bm,stages=stages)
    reference(xs[0],parts,count,bias,residual,expected,bm,stages);states.append(dict(name='refresh',**compare(out,expected,True)))
    for index,inputs in enumerate(xs):
        out.fill_(float('nan'));candidate.run(inputs,parts,count,bias,residual,out,cache,refresh=False,bm=bm,stages=stages)
        reference(inputs,parts,count,bias,residual,expected,bm,stages);states.append(dict(name=('identical','delta1pct','delta5pct','delta20pct')[index],**compare(out,expected,index==0)))
    bias.add_(.01);residual.add_(.03)
    candidate.run(xs[2],parts,count,bias,residual,out,cache,refresh=False,bm=bm,stages=stages);reference(xs[2],parts,count,bias,residual,expected,bm,stages);states.append(dict(name='fresh_bias_and_residual',**compare(out,expected)))
    wh.zero_();wl.zero_();count.zero_()
    candidate.run(xs[0],parts,count,bias,residual,out,cache,refresh=True,bm=bm,stages=stages)
    candidate.run(xs[2],parts,count,bias,residual,out,cache,refresh=False,bm=bm,stages=stages);reference(xs[2],parts,count,bias,residual,expected,bm,stages);states.append(dict(name='master_zero_with_explicit_refresh',**compare(out,expected,True)))
    wh.copy_(saved_wh);wl.copy_(saved_wl);count.fill_(int(three));bias.copy_(saved_bias);residual.copy_(saved_residual)
    candidate.run(xs[0],parts,count,bias,residual,out,cache,refresh=True,bm=bm,stages=stages)
    candidate.run(xs[2],parts,count,bias,residual,out,cache,refresh=False,bm=bm,stages=stages);reference(xs[2],parts,count,bias,residual,expected,bm,stages);states.append(dict(name='restored_master_refreshed',**compare(out,expected)))
    assert torch.equal(rng,torch.cuda.get_rng_state())
    # A full refresh plus a reuse is compared with two original GEMMs. Input
    # save, delta pack, cache read/write and both epilogues are all timed.
    b0=torch.empty_like(out);b1=torch.empty_like(out);c0=torch.empty_like(out);c1=torch.empty_like(out)
    for _ in range(3):
        reference(xs[0],parts,count,bias,residual,b0,bm,stages);reference(xs[2],parts,count,bias,residual,b1,bm,stages)
        candidate.run(xs[0],parts,count,bias,residual,c0,cache,refresh=True,bm=bm,stages=stages);candidate.run(xs[2],parts,count,bias,residual,c1,cache,refresh=False,bm=bm,stages=stages)
    baseline=torch.cuda.CUDAGraph();trial=torch.cuda.CUDAGraph();torch.cuda.synchronize()
    with torch.cuda.graph(baseline):
        reference(xs[0],parts,count,bias,residual,b0,bm,stages);reference(xs[2],parts,count,bias,residual,b1,bm,stages)
    with torch.cuda.graph(trial):
        candidate.run(xs[0],parts,count,bias,residual,c0,cache,refresh=True,bm=bm,stages=stages);candidate.run(xs[2],parts,count,bias,residual,c1,cache,refresh=False,bm=bm,stages=stages)
    for _ in range(100):baseline.replay();trial.replay()
    torch.cuda.synchronize();compare(c0,b0,True);compare(c1,b1)
    pairs=[]
    for index in range(6):
        if index%2:b=elapsed(trial);a=elapsed(baseline)
        else:a=elapsed(baseline);b=elapsed(trial)
        pairs.append(dict(baseline_two_steps_ms=a,candidate_two_steps_ms=b,saved_two_steps_ms=a-b,speedup=a/b))
    baseline.reset();trial.reset();assert torch.equal(rng,torch.cuda.get_rng_state())
    row=dict(shape=[m,n,k],BM=bm,stages=stages,weight_low_branch=three,states=states,RNG_unchanged=True,cache_bytes=sum(x.numel()*x.element_size()for x in cache.values()),all_pack_and_cache_costs_timed=True,automatic_master_owner_not_implemented=True,full_model_and_video_quality_verified=False,timing=dict(pairs=pairs,median_saved_two_steps_ms=statistics.median(x['saved_two_steps_ms']for x in pairs),all6_faster=all(x['saved_two_steps_ms']>0 for x in pairs)))
    print('DELTA_GEMM_ROW '+json.dumps(dict(shape=row['shape'],three=three,median_saved_two_steps_ms=row['timing']['median_saved_two_steps_ms'],all6_faster=row['timing']['all6_faster'])),flush=True)
    return row

@torch.no_grad()
def main():
    hold=announce_worker();torch.manual_seed(29051280);rows=[]
    for shape in ((2560,1280,5120,64,3),(640,1280,5120,32,3),(67,79,127,32,2)):
        for three in (False,True):rows.append(screen(*shape,three))
    report=dict(diagnostic_complete=True,all_dynamic_gates_passed=True,synthetic_only=True,rows=rows,current_input_always_live=True,cached_pre_bias_FP32_product=True,full_refresh_original_BK64_bitwise=True,all_pack_cache_and_refresh_costs_timed=True,full_model_and_video_quality_verified=False,official_adopted=False)
    Path('comparison.json').write_text(json.dumps(report,indent=2)+'\n');print('DELTA_GEMM_DONE rows='+str(len(rows)),flush=True)

if __name__=='__main__':main()
