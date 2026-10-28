"""Check C320 stage routing and grouped arithmetic against unchanged world."""
from unittest.mock import patch
import json
import torch
import torch.nn.functional as F
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.modules.attention import GEGLU
from unifolm_wma.utils.triton_compensated_geglu import compensated_geglu
from unifolm_wma.utils.triton_half_compensated_geglu import half_compensated_geglu
from unifolm_wma.utils.decision_half_geglu import install_decision_half_geglus, make_forward


@torch.no_grad()
def main():
    torch.manual_seed(479)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = GEGLU(320,1280).eval().cuda()
    original = lambda x: compensated_geglu(x,module.proj.weight,module.proj.bias,group_m=16)
    module.forward = original
    module._compensated_geglu_installed = True
    keys = tuple(module.state_dict())
    assert install_decision_half_geglus(module) == 1
    value = torch.randn(40960,320,device='cuda')*.7
    with patch('unifolm_wma.utils.decision_half_geglu.half_compensated_geglu',
               side_effect=AssertionError('World entered Half GEGLU')):
        torch.testing.assert_close(module(value),original(value),rtol=0,atol=0)
    torch.backends.cudnn.allow_tf32 = True
    result = module(value)
    expected = half_compensated_geglu(value,module.proj.weight,module.proj.bias,group_m=0)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)
    sample = [0,20000,40959]
    projected = F.linear(value[sample].double(),module.proj.weight.double(),module.proj.bias.double())
    v,g = projected.chunk(2,dim=-1)
    exact = v*F.gelu(g)
    error = float((result[sample].double()-exact).norm()/exact.norm())
    assert torch.isfinite(result).all() and error < 2e-6, error
    before = graph_seconds(original,value,10,3)
    seconds = graph_seconds(module,value,10,5)
    after = graph_seconds(original,value,10,3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): module(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream): captured = module(value)
    value.add_(.01)
    module.proj.weight.add_(.00001)
    module.proj.bias.mul_(.95)
    eager = module(value)
    graph.replay()
    torch.testing.assert_close(captured,eager,rtol=0,atol=0)
    torch.backends.cudnn.allow_tf32 = False
    torch.testing.assert_close(module(value),original(value),rtol=0,atol=0)
    torch.backends.cudnn.allow_tf32 = True
    for fallback in ('training','grad','autocast','noncontiguous','shape'):
        module.train(fallback=='training')
        x = value.t().contiguous().t() if fallback=='noncontiguous' else value[:35] if fallback=='shape' else value
        # A native fallback provides the full supported original GEGLU behavior.
        native = GEGLU.forward.__get__(module)
        module.forward = native
        wrapped = make_forward(module)
        with torch.set_grad_enabled(fallback=='grad'):
            with torch.autocast('cuda',enabled=fallback=='autocast',dtype=torch.float16):
                with patch('unifolm_wma.utils.decision_half_geglu.half_compensated_geglu',
                           side_effect=AssertionError('Fallback entered Half GEGLU')):
                    torch.testing.assert_close(wrapped(x),native(x),rtol=0,atol=0)
        module.eval()
    assert keys == tuple(module.state_dict())
    assert all(p.dtype==torch.float32 for p in module.parameters())
    print('DECISION_HALF_GEGLU '+json.dumps(dict(shape=[40960,320,1280],
        fp64_relative=error, grouped_vs_ungrouped_bitwise=True,world_bitwise=True,
        before_s=before,after_s=after,candidate_s=seconds,
        conservative_speedup=min(before,after)/seconds)),flush=True)
    assert min(before,after)>seconds, 'Candidate must be faster on clean GPU'
    print('DECISION_HALF_GEGLU_CHECK passed=true world_bitwise=true '
          'grouped_arithmetic_bitwise=true graph_input_weight_refresh=true '
          'native_fallbacks=5 master_weights=fp32 full_video_quality=unverified',flush=True)


if __name__=='__main__': main()
