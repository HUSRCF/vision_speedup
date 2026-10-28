"""Validate the real GEGLU wrapper, stage scope, caches and fallback paths."""
import json
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.modules.attention import GEGLU
from unifolm_wma.utils.initial_decision_linears import install_initial_decision_linears
from unifolm_wma.utils.initial_world_linears import install_initial_world_linears
from unifolm_wma.utils.decision_initial_primary import install_decision_initial_primary
from unifolm_wma.utils.nvrtc_geglu import install_nvrtc_geglus
from unifolm_wma.utils.initial_primary_geglu import install_initial_primary_geglu,make_forward


def model():
    root = torch.nn.Module()
    root.init_attn = torch.nn.Sequential(GEGLU(512,2048))
    root.other = GEGLU(512,2048)
    root.eval()
    assert install_initial_decision_linears(root)==1
    assert install_initial_world_linears(root)==1
    assert install_decision_initial_primary(root,'ffn')==1
    assert install_nvrtc_geglus(root,fmad=True,vectorized=True)==2
    return root


def exact(left,right):
    torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert torch.equal(left.contiguous().view(torch.int32),right.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(809)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    cpu = model()
    original_cpu = cpu.init_attn[0].forward
    assert install_initial_primary_geglu(cpu)==1
    xcpu = torch.randn(3,512)
    exact(cpu.init_attn[0](xcpu),original_cpu(xcpu))
    root = model().cuda()
    module = root.init_attn[0]
    original = module.forward
    keys = tuple(root.state_dict())
    assert install_initial_primary_geglu(root)==1
    assert not getattr(root.other,'_initial_primary_geglu_installed',False)
    value = torch.randn(2560,16,512,device='cuda')*.7
    torch.backends.cudnn.allow_tf32=False
    with patch('unifolm_wma.utils.initial_primary_geglu.inline_primary_geglu',
               side_effect=AssertionError('World entered decision fusion')):
        exact(module(value),original(value))
    torch.backends.cudnn.allow_tf32=True
    expected = original(value)
    exact(module(value),expected)
    before = graph_seconds(original,value,5,3)
    seconds = graph_seconds(module,value,5,3)
    after = graph_seconds(original,value,5,3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):module(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=module(value)
    value.add_(.01)
    module.proj.weight.add_(.00001)
    module.proj.bias.mul_(.95)
    eager = module(value)
    exact(eager,original(value))
    graph.replay()
    exact(captured,eager)
    for fallback in ('training','grad','autocast','noncontiguous','shape','matmul_tf32'):
        module.train(fallback=='training')
        torch.backends.cuda.matmul.allow_tf32=fallback=='matmul_tf32'
        x = (value.transpose(0,1) if fallback=='noncontiguous' else
             value[:3] if fallback=='shape' else value)
        with torch.set_grad_enabled(fallback=='grad'):
            with torch.autocast('cuda',enabled=fallback=='autocast',dtype=torch.float16):
                with patch('unifolm_wma.utils.initial_primary_geglu.inline_primary_geglu',
                           side_effect=AssertionError('Fallback entered fusion')):
                    torch.testing.assert_close(module(x),original(x),rtol=0,atol=0)
        module.eval()
        torch.backends.cuda.matmul.allow_tf32=False
    module.forward=original
    module.forward=make_forward(module)
    expected=original(value)
    stream.wait_stream(torch.cuda.current_stream())
    cold=torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold,stream=stream):cold_result=module(value)
    cold.replay()
    exact(cold_result,expected)
    assert keys==tuple(root.state_dict()) and all(p.dtype==torch.float32 for p in root.parameters())
    try:install_initial_primary_geglu(root)
    except ValueError:pass
    else:raise AssertionError('Repeated fusion installation accepted')
    print('INITIAL_PRIMARY_GEGLU_RUNTIME '+json.dumps(dict(before_s=before,after_s=after,
        candidate_s=seconds,conservative_speedup=min(before,after)/seconds)),flush=True)
    print('INITIAL_PRIMARY_GEGLU_RUNTIME_CHECK passed=true real_factory=true modules=1 '
          'outside_excluded=true world_previous_bitwise=true decision_signed_zero_bitwise=true '
          'cpu_and_native_fallbacks=7 cold_capture_fallback=true '
          'input_master_weight_bias_graph_refresh=true state_keys_master_fp32=true '
          'duplicate_rejected=true full_video_quality=unverified',flush=True)


if __name__=='__main__':
    main()
