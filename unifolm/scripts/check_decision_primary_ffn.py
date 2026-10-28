"""Measure the deliberate precision tradeoff and verify all stage/graph guards."""
from unittest.mock import patch
import json
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_compensated_linear import make_forward as precise_forward
from unifolm_wma.utils.decision_vendor_ffn import install_decision_vendor_ffns
from unifolm_wma.utils.decision_primary_ffn import install_decision_primary_ffns,make_forward,primary_linear
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight


@torch.no_grad()
def main():
    torch.manual_seed(523)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    for rows,inner,columns in [(10240,2560,640),(2560,5120,1280),(640,5120,1280)]:
        root=torch.nn.Module()
        root.block=torch.nn.Module()
        root.block.ff=torch.nn.Module()
        root.block.ff.net=torch.nn.Sequential(torch.nn.Identity(),torch.nn.Identity(),torch.nn.Linear(inner,columns))
        root.outside=torch.nn.Linear(3,3)
        root.eval().cuda()
        module=root.block.ff.net[2]
        module.forward=precise_forward(module,backend='scaled-fp16',min_matmul_macs=0,
            cache_half_weights=True,group_m=32,block_k=64,pack_inputs=True)
        module._tf32x3_installed=True
        assert install_decision_vendor_ffns(root,'chunked')==1
        original=module.forward
        keys=tuple(root.state_dict())
        assert install_decision_primary_ffns(root,'all')==1
        assert not getattr(root.outside,'_decision_primary_ffn_installed',False)
        value=torch.randn(rows,inner,device='cuda')*.7
        torch.backends.cudnn.allow_tf32=False
        with patch('unifolm_wma.utils.decision_primary_ffn.primary_linear',
                   side_effect=AssertionError('World entered primary FFN')):
            torch.testing.assert_close(module(value),original(value),rtol=0,atol=0)
        torch.backends.cudnn.allow_tf32=True
        result=module(value)
        parts=pack_half_weight(module.weight)
        reference=primary_linear(value,parts[0],module.bias,debug_poison=True)
        torch.testing.assert_close(result,reference,rtol=0,atol=0)
        idx=[0,rows//2,rows-1]
        precise=F.linear(value[idx].double(),module.weight.double(),module.bias.double())
        error=float((result[idx].double()-precise).norm()/precise.norm())
        assert torch.isfinite(result).all() and error<1e-3,error
        before=graph_seconds(original,value,10,3)
        seconds=graph_seconds(module,value,10,5)
        after=graph_seconds(original,value,10,3)
        assert min(before,after)>seconds
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):module(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=module(value)
        value.add_(.01); module.weight.add_(.00001); module.bias.mul_(.95)
        eager=module(value)
        graph.replay()
        torch.testing.assert_close(captured,eager,rtol=0,atol=0)
        for fallback in ('training','grad','autocast','noncontiguous','shape','matmul_tf32'):
            module.train(fallback=='training')
            torch.backends.cuda.matmul.allow_tf32=fallback=='matmul_tf32'
            x=value.t().contiguous().t() if fallback=='noncontiguous' else value[:35] if fallback=='shape' else value
            with torch.set_grad_enabled(fallback=='grad'):
                with torch.autocast('cuda',enabled=fallback=='autocast',dtype=torch.float16):
                    with patch('unifolm_wma.utils.decision_primary_ffn.primary_linear',
                               side_effect=AssertionError('Fallback entered primary FFN')):
                        torch.testing.assert_close(module(x),original(x),rtol=0,atol=0)
            module.eval();torch.backends.cuda.matmul.allow_tf32=False
        module.forward=original;module.forward=make_forward(module)
        cold_expected=original(value)
        cold=torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold,stream=stream):cold_result=module(value)
        cold.replay()
        torch.testing.assert_close(cold_result,cold_expected,rtol=0,atol=0)
        assert keys==tuple(root.state_dict()) and all(p.dtype==torch.float32 for p in root.parameters())
        print('DECISION_PRIMARY_FFN '+json.dumps(dict(shape=[rows,inner,columns],
            fp64_relative=error,before_s=before,after_s=after,candidate_s=seconds,
            conservative_speedup=min(before,after)/seconds,world_bitwise=True,
            poisoned_output_bitwise=True,graph_input_weight_refresh=True,
            full_video_quality='unverified')),flush=True)
        del root,module,value,result,parts,reference,precise,graph,captured,eager,cold,cold_result,cold_expected,x,original
        torch.cuda.empty_cache()
    print('DECISION_PRIMARY_FFN_CHECK passed=true actual_shapes=3 world_bitwise=true '
          'graph_input_weight_refresh=true cold_capture_fallback=true native_fallbacks=6 '
          'master_weights=fp32 full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
