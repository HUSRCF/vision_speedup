"""Real spatial/temporal attention, world preservation and QKV cache refresh."""
import json
from unittest.mock import patch
import torch
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.modules.attention import CrossAttention
from unifolm_wma.utils.triton_compensated_linear import install_compensated_linears
from unifolm_wma.utils.decision_primary_narrow import install_decision_primary_narrow
from unifolm_wma.utils.decision_primary_grouped_qkv import install_decision_primary_grouped_qkv
from unifolm_wma.utils.compensated_qkv import fused_compensated_qkv


def exact(left,right):
    assert torch.equal(left.contiguous().view(torch.int32),right.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(907)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    for temporal in [False,True]:
        root=torch.nn.Module()
        root.block=torch.nn.Module()
        module=CrossAttention(320,heads=5,dim_head=64,
            temporal_length=16 if temporal else None).eval().cuda()
        root.block.attn=module
        root.eval()
        module._inference_attention_layout='bmhk'
        module._compensated_spatial_attention=not temporal
        module._inference_fused_temporal_attention_fp32=temporal
        install_compensated_linears(root,backend='scaled-fp16',cache_half_weights=True,
            group_m=32,block_k=64,pack_inputs=True,narrow_output=True)
        assert install_decision_primary_narrow(root,'all')==4
        keys=tuple(root.state_dict())
        value=torch.randn(2560,16,320,device='cuda')*.7 if temporal else torch.randn(16,2560,320,device='cuda')*.7
        torch.backends.cudnn.allow_tf32=False
        world=module(value)
        torch.backends.cudnn.allow_tf32=True
        expected=module(value)
        before=graph_seconds(module,value,5,3)
        if not temporal:assert install_decision_primary_grouped_qkv(root)==0
        assert install_decision_primary_grouped_qkv(root,scope='all')==1
        actual=module(value)
        exact(actual,expected)
        seconds=graph_seconds(module,value,5,3)
        module._compensated_self_qkv_enabled=False
        after=graph_seconds(module,value,5,3)
        module._compensated_self_qkv_enabled=True
        torch.backends.cudnn.allow_tf32=False
        with patch('unifolm_wma.utils.decision_primary_grouped_qkv.primary_grouped_qkv',
                   side_effect=AssertionError('World entered primary grouped QKV')):
            exact(module(value),world)
        torch.backends.cudnn.allow_tf32=True
        module(value)
        cache=module._primary_grouped_qkv_cache
        ptrs=[cache[1].data_ptr(),*(p.data_ptr() for p in cache[2])]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):module(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=module(value)
        value.add_(.001)
        for p in (module.to_q,module.to_k,module.to_v):p.weight.add_(.00001)
        eager=module(value)
        cache=module._primary_grouped_qkv_cache
        assert ptrs==[cache[1].data_ptr(),*(p.data_ptr() for p in cache[2])]
        graph.replay();exact(captured,eager)
        module._compensated_self_qkv_enabled=False
        exact(eager,module(value))
        module._compensated_self_qkv_enabled=True
        for fallback in ['training','grad','autocast','noncontiguous','shape','matmul_tf32']:
            module.train(fallback=='training')
            torch.backends.cuda.matmul.allow_tf32=fallback=='matmul_tf32'
            x=value.transpose(0,1) if fallback=='noncontiguous' else value[:3] if fallback=='shape' else value
            with torch.set_grad_enabled(fallback=='grad'):
                with torch.autocast('cuda',enabled=fallback=='autocast',dtype=torch.float16):
                    with patch('unifolm_wma.utils.decision_primary_grouped_qkv.primary_grouped_qkv',
                               side_effect=AssertionError('Fallback entered primary grouped QKV')):
                        result=fused_compensated_qkv(module,x)
                        for part,p in zip(result,(module.to_q,module.to_k,module.to_v)):
                            torch.testing.assert_close(part,p(x),rtol=0,atol=0)
            module.eval();torch.backends.cuda.matmul.allow_tf32=False
        del module._primary_grouped_qkv_cache
        # Original individual projection caches are warm; the merged cache must stay cold.
        stream.wait_stream(torch.cuda.current_stream())
        cold=torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold,stream=stream):cold_output=fused_compensated_qkv(module,value)
        cold.replay()
        assert not hasattr(module,'_primary_grouped_qkv_cache')
        for part,p in zip(cold_output,(module.to_q,module.to_k,module.to_v)):exact(part,p(value))
        assert keys==tuple(root.state_dict()) and all(p.dtype==torch.float32 for p in root.parameters())
        try:install_decision_primary_grouped_qkv(root,scope='all')
        except ValueError:pass
        else:raise AssertionError('Duplicate installation accepted')
        print('PRIMARY_GROUPED_QKV_RUNTIME '+json.dumps(dict(temporal=temporal,before_s=before,
            after_s=after,candidate_s=seconds,conservative_speedup=min(before,after)/seconds)),flush=True)
    print('PRIMARY_GROUPED_QKV_RUNTIME_CHECK passed=true real_attention_modules=2 '
          'spatial_temporal_outputs_signed_zero_bitwise=true world_previous_bitwise=true '
          'input_three_master_weights_graph_refresh=true concat_and_parts_storage_reused=true '
          'cold_capture_native_fallbacks=true state_keys_fp32_masters=true '
          'duplicate_rejected=true full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
