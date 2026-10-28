"""Compare against current original GEGLU/three-product FFN mathematics."""
import json
import os
import statistics
import torch
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_zero_weight_low_geglu import residual_count
from unifolm_wma.utils.geglu_from_input_parts import geglu_from_input_parts
from unifolm_wma.utils.triton_prepacked_input_linear import prepacked_input_linear
from unifolm_wma.utils.geglu_hidden_parts_c1280_pipeline import geglu_hidden_parts_c1280
from unifolm_wma.utils.ffn1280_zero_residual import shared_final_low_count,zero_ffn1280_hidden_residual
from unifolm_wma.utils import geglu_output_allocation,gemm_output_allocation,half_pack_allocation


def exact(left,right):
    assert left.shape==right.shape and left.dtype==right.dtype
    bits=torch.int16 if left.dtype==torch.float16 else torch.int32
    assert torch.equal(left.contiguous().view(bits),right.contiguous().view(bits))


@torch.no_grad()
def main():
    torch.manual_seed(1407);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU='all',UNIFOLM_GEMM_OUTPUT_NOFILL='all',
        UNIFOLM_HALF_PACK_NOFILL='1',UNIFOLM_GEGLU_OUTPUT_NOFILL='1')
    fixtures=0
    for index,(rows,channels) in enumerate([(2560,1280),(640,1280),(13,71)]):
        for bias in [False,True]:
            for quantized in [True,False]:
                value=torch.randn(rows,1280,device='cuda')*.3
                residual=torch.randn(rows,channels,device='cuda')*.3
                gw=torch.randn(10240,1280,device='cuda')*.02
                fw=torch.randn(channels,5120,device='cuda')*.02
                if quantized:gw.copy_(gw.half().float());fw.copy_(fw.half().float())
                gb=torch.randn(10240,device='cuda')*.02 if bias else None
                fb=torch.randn(channels,device='cuda')*.02 if bias else None
                gp=pack_half_weight(gw);gc=residual_count(gp,refresh=True)
                fp=pack_half_weight(fw);fc=shared_final_low_count(fp)
                assert (int(fc)==0)==quantized
                storage=tuple(p.data_ptr() for p in (*gp,*fp)),gc.data_ptr(),fc.data_ptr()
                def baseline(x):
                    inputs=pack_half_weight(x,weight_scale=1.)
                    hidden=geglu_from_input_parts(inputs,gp,gc,gb)
                    return prepacked_input_linear(hidden,fw,fb,weight_parts=fp,
                        block_m=64 if rows>=2560 else 32,block_n=64,block_k=64,
                        group_m=32,num_warps=4,num_stages=2)+residual
                def candidate(x):
                    inputs=pack_half_weight(x,weight_scale=1.)
                    hidden=geglu_hidden_parts_c1280(inputs,gp,gc,gb)
                    return zero_ffn1280_hidden_residual(hidden,fp,fc,residual,fb)
                inputs=pack_half_weight(value,weight_scale=1.)
                hidden=geglu_from_input_parts(inputs,gp,gc,gb)
                expected_parts=pack_half_weight(hidden,weight_scale=1.)
                parts=geglu_hidden_parts_c1280(inputs,gp,gc,gb)
                for a,b in zip(parts,expected_parts):exact(a,b)
                fresh=geglu_hidden_parts_c1280(inputs,gp,gc,gb)
                assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(parts,fresh))
                reference=baseline(value);exact(reference,candidate(value))
                policies=[geglu_output_allocation,gemm_output_allocation,half_pack_allocation]
                previous=[p._debug_poison for p in policies]
                try:
                    for p in policies:p._debug_poison=float('nan')
                    poisoned=candidate(value)
                finally:
                    for p,v in zip(policies,previous):p._debug_poison=v
                assert torch.isfinite(poisoned).all();exact(reference,poisoned)
                timing=None
                if index<2:
                    for fn in [baseline,candidate]:graph_seconds(fn,value,5,2)
                    pairs=[]
                    for iteration in range(6):
                        order=[('baseline',baseline),('candidate',candidate)]
                        if iteration%2:order.reverse()
                        pairs.append({label:graph_seconds(fn,value,5,3) for label,fn in order})
                    seconds={k:statistics.median(p[k] for p in pairs) for k in ['baseline','candidate']}
                    timing=dict(seconds=seconds,speedup=seconds['baseline']/seconds['candidate'],pairs=pairs)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):candidate(value)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):captured=candidate(value)
                for q in [False,True]:
                    value.add_(.001);residual.mul_(.97);gw.add_(.00013);fw.add_(.00017)
                    if q:gw.copy_(gw.half().float());fw.copy_(fw.half().float())
                    if gb is not None:gb.add_(.0002);fb.add_(.0003)
                    pack_half_weight(gw,out=gp);refreshed=residual_count(gp,refresh=True)
                    pack_half_weight(fw,out=fp);final_refreshed=shared_final_low_count(fp,refresh=True)
                    assert refreshed is gc and final_refreshed is fc
                    assert (int(gc)==0)==q and (int(fc)==0)==q
                    assert storage==(tuple(p.data_ptr() for p in (*gp,*fp)),gc.data_ptr(),fc.data_ptr())
                    graph.replay();exact(captured,baseline(value))
                print('FFN1280_ZERO_RESIDUAL_MICRO '+json.dumps(dict(index=index,rows=rows,channels=channels,
                    bias=bias,quantized=quantized,hidden_and_final_signedzero=True,nan_all_buffers=True,
                    fresh_hidden_parts=True,mutable_graph_input_residual_GE_FFN_masters_bias=True,
                    zero_nonzero_zero_both_predicates=True,shared_static_parts_count_storage=True,
                    original_short_BK64_arithmetic=True,timing=timing,inference_installed=False)),flush=True)
                fixtures+=1
                graph.reset()
                del value,residual,gw,fw,gb,fb,gp,gc,fp,fc,inputs,hidden,expected_parts,parts,fresh,reference,poisoned,graph,captured
                torch.cuda.empty_cache()
    assert fixtures==12
    print('FFN1280_ZERO_RESIDUAL_MICRO_CHECK passed=true fixtures=12 signedzero=true NaN=true mutable_graph=true',flush=True)


if __name__=='__main__':main()
