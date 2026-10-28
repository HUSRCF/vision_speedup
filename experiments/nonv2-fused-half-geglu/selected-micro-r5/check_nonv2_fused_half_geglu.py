"""Checkpoint-backed synthetic normalized inputs; no full-video quality claim."""
import json
import os
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
from unifolm_wma.utils.triton_geglu import fused_geglu
from nonv2_fused_half_geglu import fused


CONFIGS=[(64,64,32,4,3),(128,64,32,8,3),(64,128,32,8,3),(128,64,64,8,3),(64,128,64,8,3)]


def errors(a,b):
    assert a.dtype==b.dtype==torch.float16 and a.shape==b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    x,y=a.float(),b.float();d=x-y
    return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/y.norm().clamp_min(1e-30)),
                mse=float(d.square().mean()),signedzero_bits=torch.equal(a.view(torch.int16),b.view(torch.int16)))


@torch.no_grad()
def main():
    torch.manual_seed(1570)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    checkpoint=torch.load('/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',map_location='cpu',mmap=True)
    state=checkpoint.get('state_dict',checkpoint)
    selected=os.environ.get('EXPERIMENT_SWEEP_SCOPE')=='selected320'
    geometries=[(40960,320)] if selected else [(40960,320),(10240,640),(2560,1280),(40960,512)]
    configs=[(64,128,32,8,3)] if selected else CONFIGS
    records=[]
    for m,c in geometries:
        names=sorted(n for n,t in state.items() if n.endswith('.ff.net.0.proj.weight') and n.startswith('model.diffusion_model.') and tuple(t.shape)==(8*c,c))
        assert names,(m,c)
        name=names[0];w=state[name].half().cuda().contiguous();b=state[name[:-6]+'bias'].half().cuda().contiguous()
        x=torch.randn(m,c,device='cuda',dtype=torch.float32).half();saved=x.clone();saved_w=w.clone();saved_b=b.clone()
        rng=torch.cuda.get_rng_state().clone();pointers=[t.data_ptr()for t in (x,w,b)]
        baseline=lambda:fused_geglu(F.linear(x,w,b))
        expected=baseline()
        for config in configs:
            # Never start another compiler variant when the bounded metadata
            # budget cannot accommodate compiler intermediates and evidence.
            import shutil
            assert shutil.disk_usage(os.environ['TRITON_CACHE_DIR']).free>512*1024
            assert shutil.disk_usage('/tmp').free>512*1024
            actual=fused(x,w,b,config);initial_error=errors(actual,expected);del actual
            candidate=lambda:fused(x,w,b,config)
            graphs={};outputs={}
            for label,fn in [('baseline',baseline),('candidate',candidate)]:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):out=fn()
                torch.cuda.current_stream().wait_stream(stream);del out
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):outputs[label]=fn()
                torch.cuda.current_stream().wait_stream(stream);graphs[label]=graph
            cases=[]
            for label in ('initial','changed_inputs','changed_weights','restored','zero_weights','zero_then_restore'):
                if label=='changed_inputs':x.add_(.001)
                elif label=='changed_weights':w.add_(.00013);b.add_(.00013)
                elif label=='zero_weights':w.zero_();b.zero_()
                elif label in ('restored','zero_then_restore'):x.copy_(saved);w.copy_(saved_w);b.copy_(saved_b)
                for graph in graphs.values():graph.replay()
                torch.cuda.synchronize();err=errors(outputs['candidate'],outputs['baseline'])
                if label in ('restored','zero_then_restore'):
                    assert torch.equal(outputs['baseline'].view(torch.int16),expected.view(torch.int16))
                cases.append(dict(case=label,**err))
            assert torch.equal(rng,torch.cuda.get_rng_state()) and pointers==[t.data_ptr()for t in (x,w,b)]
            for t,original in zip((x,w,b),(saved,saved_w,saved_b)):assert torch.equal(t.view(torch.int16),original.view(torch.int16))
            for i in range(30):graphs['candidate' if i%2 else 'baseline'].replay()
            torch.cuda.synchronize();pairs=[]
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for i in range(6):
                pair={}
                for label in (('baseline','candidate')if i%2==0 else('candidate','baseline')):
                    for _ in range(3):graphs[label].replay()
                    samples=[]
                    for _ in range(7):
                        start.record();graphs[label].replay();end.record();end.synchronize();samples.append(start.elapsed_time(end))
                    pair[label+'_ms']=statistics.median(samples);pair[label+'_samples_ms']=samples
                pair['saved_ms']=pair['baseline_ms']-pair['candidate_ms'];pairs.append(pair)
            row=dict(M=m,C=c,H=4*c,checkpoint_tensor_name=name,config=list(config),initial_error=initial_error,cases=cases,pairs=pairs,
                     median_saved_ms=statistics.median(p['saved_ms']for p in pairs),all6_pairs_positive=all(p['saved_ms']>0 for p in pairs),
                     input_weight_bias_changed_and_restored=True,RNG_unchanged=True,original_values_and_pointers_restored=True)
            records.append(row);print('FUSED_HALF_GEGLU_ROW '+json.dumps(row),flush=True)
            for graph in graphs.values():graph.reset()
            del graphs,outputs
        del x,w,b,saved,saved_w,saved_b,expected
    result=dict(passed=True,records=records,selected320_only=selected,checkpoint_backed=True,synthetic_LayerNorm_like_inputs=True,
                baseline_is_cublas_Half_linear_plus_adopted_fused_geglu=True,numerical_candidate=True,
                no_full_model_or_video_quality_claim=True,no_official_score=True)
    Path(os.environ['EXPERIMENT_REPORT']).write_text(json.dumps(result,indent=2)+'\n')
    print('NONV2_FUSED_HALF_GEGLU_MICRO passed=true',flush=True)


if __name__=='__main__':main()
