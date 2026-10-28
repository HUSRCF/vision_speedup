"""Include the unchanged original GEGLU in the residual norm fusion timing."""
import json
import os
import statistics
import torch
from benchmark_temporal_conv2d import graph_seconds
from benchmark_residual_norm_half_pack import exact
from unifolm_wma.utils.triton_residual_layer_norm import fused_residual_layer_norm
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_zero_weight_low_geglu import residual_count
from unifolm_wma.utils.triton_world_narrow_geglu import primary_narrow_geglu
from unifolm_wma.utils.triton_branch_protected_geglu import branch_protected_geglu
from unifolm_wma.utils.triton_prepacked_geglu import prepacked_geglu
from unifolm_wma.utils.residual_norm_half_pack import residual_norm_half_pack
from unifolm_wma.utils.geglu_from_input_parts import geglu_from_input_parts
from unifolm_wma.utils import norm_buffer_allocation,half_pack_allocation,geglu_output_allocation


@torch.no_grad()
def main():
    torch.manual_seed(726);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU='all',UNIFOLM_NORM_BUFFER_NOFILL='all',
                     UNIFOLM_HALF_PACK_NOFILL='1',UNIFOLM_GEGLU_OUTPUT_NOFILL='1')
    fixtures=[(40960,320),(10240,640),(2560,1280),(640,1280),(73,320),(17,640)]
    for index,(rows,inner) in enumerate(fixtures):
        for quantized in [True,False]:
            channels=inner*4 if index<4 else 59
            left=torch.randn(rows,inner,device='cuda')*.7;right=torch.randn_like(left)*.3
            norm=torch.nn.LayerNorm(inner,eps=1e-5).cuda().eval()
            norm.weight.uniform_(.7,1.3);norm.bias.uniform_(-.2,.2)
            weight=torch.randn(channels*2,inner,device='cuda')*.02
            if quantized:weight.copy_(weight.half().float())
            bias=torch.randn(channels*2,device='cuda')*.02 if index%2==0 else None
            parts=pack_half_weight(weight);count=residual_count(parts)
            pointers=tuple(p.data_ptr() for p in parts),count.data_ptr()
            def original_geglu(x):
                if inner==320:return primary_narrow_geglu(x,parts,bias,mode='gate',block_n=64)
                if inner==640:return branch_protected_geglu(x,parts,bias,mode='gate',block_n=64,group_m=32)
                return prepacked_geglu(x,parts,bias,block_m=64,block_n=32,group_m=32,num_warps=4,num_stages=2)
            def baseline(x):
                residual,normalized=fused_residual_layer_norm(x,right,norm)
                return residual,original_geglu(normalized)
            def candidate(x):
                residual,high,low=residual_norm_half_pack(x,right,norm)
                return residual,geglu_from_input_parts((high,low),parts,count,bias)
            reference=baseline(left);actual=candidate(left);exact(reference,actual)
            modules=[norm_buffer_allocation,half_pack_allocation,geglu_output_allocation]
            previous=[m._debug_poison for m in modules]
            try:
                for m in modules:m._debug_poison=float('nan')
                poisoned=candidate(left)
            finally:
                for m,v in zip(modules,previous):m._debug_poison=v
            assert all(torch.isfinite(x).all() for x in poisoned);exact(reference,poisoned)
            timing=None
            if index<4:
                for f in [baseline,candidate]:graph_seconds(f,left,5,2)
                pairs=[]
                for iteration in range(6):
                    order=[('baseline',baseline),('candidate',candidate)]
                    if iteration%2:order.reverse()
                    pairs.append({label:graph_seconds(fn,left,5,3) for label,fn in order})
                seconds={label:statistics.median(r[label] for r in pairs) for label in ['baseline','candidate']}
                timing=dict(seconds=seconds,speedup=seconds['baseline']/seconds['candidate'],pairs=pairs)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):candidate(left)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):captured=candidate(left)
            for next_quantized in [not quantized,quantized]:
                left.add_(.001);right.mul_(.97);norm.weight.mul_(1.001);norm.bias.add_(.0003)
                weight.add_(.00013)
                if next_quantized:weight.copy_(weight.half().float())
                if bias is not None:bias.add_(.0002)
                pack_half_weight(weight,out=parts);residual_count(parts,refresh=True)
                assert pointers==(tuple(p.data_ptr() for p in parts),count.data_ptr())
                assert (int(count)==0)==next_quantized
                graph.replay();exact(captured,baseline(left))
            print('NORM3_GEGLU_PIPELINE '+json.dumps(dict(index=index,shape=[rows,inner,channels],quantized=quantized,
                residual_hidden_signedzero=True,all_buffers_nan_poison=True,
                graph_changed_both_inputs_norm_weight_bias_and_geglu_weight_bias=True,
                original_geglu_JIT_BK64_BM_BN_group_and_primary_modes=True,
                count_and_weight_parts_same_storage=True,timing=timing,inference_installed=False)),flush=True)
            del left,right,norm,weight,bias,parts,count,reference,actual,poisoned,captured,graph
            torch.cuda.empty_cache()
    print('NORM3_GEGLU_PIPELINE_CHECK passed=true fixtures=12 signedzero=true nan_poison=true '
          'graph_inputs_norm_and_geglu_masters_predicate=true original_geglu_JIT=true inference_installed=false',flush=True)


if __name__=='__main__':main()
