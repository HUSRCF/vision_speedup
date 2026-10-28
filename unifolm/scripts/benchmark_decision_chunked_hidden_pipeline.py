"""Measure C640 norm->both-primary GEGLU->chunk512 compensated FFN fusion."""
import json
import os
import statistics
import torch
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_primary_geglu import primary_geglu
from unifolm_wma.utils.cublas_chunked_compensated_linear import chunked_compensated_linear
from unifolm_wma.utils.triton_residual_layer_norm import fused_residual_layer_norm
from unifolm_wma.utils.residual_norm_half_pack import residual_norm_half_pack
from unifolm_wma.utils.decision_chunked_hidden_pipeline import primary_geglu_hidden_parts,chunked_hidden_residual
from unifolm_wma.utils import geglu_output_allocation as ga,half_pack_allocation as ha,norm_buffer_allocation as na


def exact(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape
    bits=torch.int16 if a.dtype==torch.float16 else torch.int32
    assert torch.equal(a.contiguous().view(bits),b.contiguous().view(bits)),dict(max_abs=float((a-b).abs().max()))


@torch.no_grad()
def main():
    torch.manual_seed(1137);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(UNIFOLM_GEMM_OUTPUT_NOFILL='all',UNIFOLM_HALF_PACK_NOFILL='1',UNIFOLM_GEGLU_OUTPUT_NOFILL='1',UNIFOLM_NORM_BUFFER_NOFILL='all')
    for index,(rows,channels) in enumerate([(10240,640),(640,640),(13,71)]):
        for bias in [False,True]:
            for quantized in [False,True]:
                left=torch.randn(rows,640,device='cuda')*.3;right=torch.randn_like(left)*.3
                norm=torch.nn.LayerNorm(640).cuda().eval();norm.weight.mul_(1.003);norm.bias.add_(.001)
                gw=torch.randn(5120,640,device='cuda')*.02;fw=torch.randn(channels,2560,device='cuda')*.02
                if quantized:gw.copy_(gw.half().float());fw.copy_(fw.half().float())
                gb=torch.randn(5120,device='cuda')*.02 if bias else None;fb=torch.randn(channels,device='cuda')*.02 if bias else None
                gp=pack_half_weight(gw);fp=pack_half_weight(fw);pointers=tuple(p.data_ptr() for p in (*gp,*fp))
                def baseline(value):
                    residual,normalized=fused_residual_layer_norm(value,right,norm)
                    hidden=primary_geglu(normalized,gp,gb,mode='both')
                    output=chunked_compensated_linear(hidden,fp,fb,chunk_k=512)
                    assert output is not None
                    return residual[:,:channels].contiguous()+output
                def candidate(value,poison=False):
                    residual,high,low=residual_norm_half_pack(value,right,norm)
                    hidden=primary_geglu_hidden_parts((high,low),gp,gb)
                    result=chunked_hidden_residual(hidden,fp,residual[:,:channels].contiguous(),fb,
                                                   debug_fill=float('nan') if poison else None)
                    assert result is not None
                    return result
                ref_res,normalized=fused_residual_layer_norm(left,right,norm)
                residual,high,low=residual_norm_half_pack(left,right,norm);exact(residual,ref_res)
                hidden=primary_geglu_hidden_parts((high,low),gp,gb)
                expected=pack_half_weight(primary_geglu(normalized,gp,gb,mode='both'),weight_scale=1.)
                for a,b in zip(hidden,expected):exact(a,b)
                fresh=primary_geglu_hidden_parts((high,low),gp,gb)
                assert len({p.data_ptr() for p in (*hidden,*fresh)})==4
                reference=baseline(left);actual=candidate(left);exact(actual,reference)
                policies=[ga,ha,na];previous=[p._debug_poison for p in policies]
                try:
                    for p in policies:p._debug_poison=float('nan')
                    poisoned=candidate(left,poison=True)
                finally:
                    for p,v in zip(policies,previous):p._debug_poison=v
                exact(poisoned,reference);assert torch.isfinite(poisoned).all()
                timing=None
                if index<2:
                    for fn in [baseline,candidate]:graph_seconds(fn,left,5,2)
                    pairs=[]
                    for iteration in range(6):
                        order=[('baseline',baseline),('candidate',candidate)]
                        if iteration%2:order.reverse()
                        pairs.append({label:graph_seconds(fn,left,5,3) for label,fn in order})
                    seconds={k:statistics.median(p[k] for p in pairs) for k in ['baseline','candidate']}
                    timing=dict(seconds=seconds,speedup=seconds['baseline']/seconds['candidate'],pairs=pairs)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):candidate(left)
                torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):captured=candidate(left)
                for q in [False,True]:
                    left.add_(.001);right.mul_(.97);norm.weight.mul_(1.001);norm.bias.add_(.0002)
                    gw.add_(.00013);fw.add_(.00013)
                    if q:gw.copy_(gw.half().float());fw.copy_(fw.half().float())
                    if gb is not None:gb.add_(.0001);fb.add_(.0001)
                    pack_half_weight(gw,out=gp);pack_half_weight(fw,out=fp)
                    assert pointers==tuple(p.data_ptr() for p in (*gp,*fp))
                    graph.replay();exact(captured,baseline(left))
                print('DECISION_CHUNKED_HIDDEN_PIPELINE '+json.dumps(dict(index=index,shape=[rows,640,2560,channels],bias=bias,quantized=quantized,
                    residual_hidden_Half_and_final_signedzero=True,fresh_hidden_each_call=True,nan_poison_all_dynamic_and_cublas_output=True,
                    graph_both_inputs_norm_GEGLU_final_masters_bias=True,static_parts_storage_reused=True,
                    original_primary_GEGLU_and_chunk512_batched_GEMM_ordered_reduce_math=True,timing=timing,inference_installed=False)),flush=True)
                del left,right,norm,gw,fw,gb,fb,gp,fp,ref_res,normalized,residual,high,low,hidden,expected,fresh,reference,actual,poisoned,graph,captured
                torch.cuda.empty_cache()
    print('DECISION_CHUNKED_HIDDEN_PIPELINE_CHECK passed=true fixtures=12 original_math=true no_model_install=true',flush=True)


if __name__=='__main__':main()
