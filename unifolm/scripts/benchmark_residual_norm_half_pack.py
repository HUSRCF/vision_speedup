"""Compare exact residual LayerNorm+Half parts including all per-call work."""
import json
import os
import statistics
import torch
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_residual_layer_norm import fused_residual_layer_norm
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.residual_norm_half_pack import residual_norm_half_pack
from unifolm_wma.utils import norm_buffer_allocation,half_pack_allocation


def exact(a,b):
    assert len(a)==len(b)
    for left,right in zip(a,b):
        assert left.dtype==right.dtype and left.shape==right.shape
        dtype=torch.int32 if left.dtype==torch.float32 else torch.int16
        assert torch.equal(left.contiguous().view(dtype),right.contiguous().view(dtype))


@torch.no_grad()
def main():
    torch.manual_seed(725);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(UNIFOLM_NORM_BUFFER_NOFILL='all',UNIFOLM_HALF_PACK_NOFILL='1')
    fixtures=[(40960,320),(10240,640),(2560,1280),(640,1280),(73,321),(17,96),(1,320),(2,640)]
    for index,(rows,channels) in enumerate(fixtures):
        norm=torch.nn.LayerNorm(channels,eps=1e-5 if index%2 else 1e-6).cuda().eval()
        norm.weight.uniform_(.7,1.3);norm.bias.uniform_(-.2,.2)
        left=torch.randn(rows,channels,device='cuda')*.7
        right=torch.randn_like(left)*.3
        def baseline(x,debug=False):
            residual,normalized=fused_residual_layer_norm(x,right,norm)
            parts=pack_half_weight(normalized.view(rows,channels),weight_scale=1.)
            return (residual,*parts,normalized) if debug else (residual,*parts)
        candidate=lambda x:residual_norm_half_pack(x,right,norm)
        reference=baseline(left,True)
        actual=residual_norm_half_pack(left,right,norm,debug=True)
        exact(reference,actual)
        modules=[norm_buffer_allocation,half_pack_allocation]
        previous=[m._debug_poison for m in modules]
        try:
            for m in modules:m._debug_poison=float('nan')
            poisoned=residual_norm_half_pack(left,right,norm,debug=True)
            poisoned_fast=candidate(left)
        finally:
            for m,v in zip(modules,previous):m._debug_poison=v
        assert all(torch.isfinite(x).all() for x in [*poisoned,*poisoned_fast])
        exact(reference,poisoned);exact(reference[:3],poisoned_fast)
        timing=None
        if index<4:
            for f in [baseline,candidate]:graph_seconds(f,left,20,2)
            pairs=[]
            for iteration in range(6):
                order=[('baseline',baseline),('candidate',candidate)]
                if iteration%2:order.reverse()
                pairs.append({label:graph_seconds(fn,left,20,3) for label,fn in order})
            seconds={label:statistics.median(r[label] for r in pairs) for label in ['baseline','candidate']}
            timing=dict(seconds=seconds,speedup=seconds['baseline']/seconds['candidate'],pairs=pairs)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):candidate(left)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=candidate(left)
        for _ in range(2):
            left.add_(.001);right.mul_(.97);norm.weight.mul_(1.001);norm.bias.add_(.0003)
            graph.replay();exact(captured,baseline(left))
        print('RESIDUAL_NORM_HALF_PACK '+json.dumps(dict(index=index,shape=[rows,channels],eps=norm.eps,
            residual_normalized_and_half_parts_signedzero=True,fast_path_half_parts_signedzero=True,
            all_buffers_nan_poison=True,graph_changed_both_inputs_norm_master_bias=True,
            timing=timing,inference_installed=False)),flush=True)
        del norm,left,right,reference,actual,poisoned,poisoned_fast,captured,graph
        torch.cuda.empty_cache()
    print('RESIDUAL_NORM_HALF_PACK_CHECK passed=true fixtures=8 original_norm_math=true '
          'all_parts_signedzero=true nan_poison=true graph_changed_inputs_masters=true inference_installed=false',flush=True)


if __name__=='__main__':main()
