"""Check the static cache and fallback contract, then real factory membership."""
import json
import os
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.utils import world_qkv_packed_attention as helper
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_zero_weight_low_linear import zero_weight_low_linear,weight_low_count
from unifolm_wma.utils.triton_prepacked_compensated_attention import prepacked_compensated_attention


def exact(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape
    assert torch.equal(a.contiguous().view(torch.int32 if a.dtype==torch.float32 else torch.int16),
                       b.contiguous().view(torch.int32 if b.dtype==torch.float32 else torch.int16))


def configure(module):
    module._compensated_spatial_attention=True
    module._world_zero_weight_low_qkv_installed=True
    module._compensated_self_qkv_enabled=True
    module._inference_attention_layout='bmhk'
    module._inference_attention_dtype=torch.float32
    module._inference_projection_dtype=torch.float32


class Fixture(torch.nn.Module):
    def __init__(self,channels,bias):
        super().__init__()
        self.heads=channels//64;self.dim_head=64;self.scale=.125;self.relative_position=False
        self.to_q=torch.nn.Linear(channels,channels,bias=False,device='cuda')
        self.to_k=torch.nn.Linear(channels,channels,bias=False,device='cuda')
        self.to_v=torch.nn.Linear(channels,channels,bias=False,device='cuda')
        self.to_out=torch.nn.Linear(channels,channels,bias=bias,device='cuda')
        configure(self);self.eval()

    def forward(self,value,context=None,mask=None):
        # Dispatch sentinel: the actual accepted-model path is checked by the
        # separate both-stage graph pair; this fixture checks wrapper guards.
        return value*.75+.25


def reference(module,value):
    batch,length,channels=value.shape
    combined=torch.cat([p.weight for p in [module.to_q,module.to_k,module.to_v]],dim=0)
    parts=pack_half_weight(combined);count=weight_low_count(parts)
    q,k,v=zero_weight_low_linear(value,parts,count,output_groups=3).unbind(0)
    output=prepacked_compensated_attention(*(x.view(batch,length,channels//64,64) for x in [q,k,v]),block_m=64,block_n=64)
    return module.to_out(output.view_as(value))


@torch.no_grad()
def main():
    torch.manual_seed(831)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(UNIFOLM_GEMM_OUTPUT_NOFILL='all',UNIFOLM_HALF_PACK_NOFILL='1',UNIFOLM_ATTENTION_OUTPUT_NOFILL='all')
    for channels,length in [(320,2560),(640,640)]:
        for bias in [False,True]:
            module=Fixture(channels,bias)
            for p in [module.to_q,module.to_k,module.to_v]:p.weight.copy_(p.weight.half().float())
            masters=tuple(p.weight.data_ptr() for p in [module.to_q,module.to_k,module.to_v,module.to_out])
            keys=set(module.state_dict());old=module.forward;forward=helper.make_forward(module)
            value=torch.randn(16,length,channels,device='cuda')*.3
            packs=[];pack=helper.pack_half_weight
            def recorded(*values,**options):
                result=pack(*values,**options);packs.append(tuple(p.data_ptr() for p in result));return result
            with patch.object(helper,'pack_half_weight',side_effect=recorded):
                actual=forward(value);exact(actual,reference(module,value));assert len(packs)==1
                cache=module._world_zero_weight_low_qkv_cache
                pointers=cache[1].data_ptr(),tuple(p.data_ptr() for p in cache[2]),cache[3].data_ptr()
                for _ in range(3):exact(actual,forward(value))
                assert len(packs)==1
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):forward(value)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):captured=forward(value)
                for quantized in [False,True]:
                    value.add_(.0001)
                    for p in [module.to_q,module.to_k,module.to_v]:
                        p.weight.add_(.00013)
                        if quantized:p.weight.copy_(p.weight.half().float())
                    module.to_out.weight.add_(.00001)
                    if module.to_out.bias is not None:module.to_out.bias.add_(.0001)
                    eager=forward(value);cache=module._world_zero_weight_low_qkv_cache
                    assert pointers==(cache[1].data_ptr(),tuple(p.data_ptr() for p in cache[2]),cache[3].data_ptr())
                    assert (int(cache[3])==0)==quantized
                    graph.replay();exact(captured,eager);exact(captured,reference(module,value))
                assert len(packs)==3
            assert keys==set(module.state_dict())
            assert masters==tuple(p.weight.data_ptr() for p in [module.to_q,module.to_k,module.to_v,module.to_out])
            assert all(p.dtype==torch.float32 for p in module.parameters())
            with patch.object(helper,'qkv_packed_attention',side_effect=AssertionError('fallback entered new kernel')):
                for flag in ['cudnn','matmul']:
                    target=torch.backends.cudnn if flag=='cudnn' else torch.backends.cuda.matmul
                    target.allow_tf32=True;exact(forward(value),old(value));target.allow_tf32=False
                module.train();exact(forward(value),old(value));module.eval()
                with torch.enable_grad():exact(forward(value),old(value))
                with torch.autocast('cuda',dtype=torch.float16):exact(forward(value),old(value))
                for other in [value[:1],value.half(),value.transpose(1,2).contiguous().transpose(1,2),value.cpu()]:
                    exact(forward(other),old(other))
                exact(forward(value,context=value),old(value,context=value))
                mask=torch.ones(1,device='cuda');exact(forward(value,mask=mask),old(value,mask=mask))
            cold=Fixture(channels,bias);cold_original=cold.forward;cold_forward=helper.make_forward(cold)
            with patch.object(helper,'pack_half_weight',side_effect=AssertionError('cold cache initialized')):
                coldgraph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(coldgraph):coldout=cold_forward(value)
                coldgraph.replay();exact(coldout,cold_original(value))
            assert not hasattr(cold,'_world_zero_weight_low_qkv_cache')
            print('QKV_PACKED_ATTENTION_RUNTIME '+json.dumps(dict(channels=channels,bias=bias,
                reference_output_signedzero=True,cache_first_warm_once=True,zero_nonzero_zero_predicate_graph=True,
                changed_input_three_masters_output_master_bias_graph=True,cache_same_storage=True,
                fp32_masters_state_unchanged=True,native_fallbacks=11,cold_capture_fallback=True,
                fallback_fixture_is_dispatch_sentinel=True)),flush=True)
            del module,forward,old,value,actual,captured,eager,cache,graph,cold,cold_forward,cold_original,coldgraph,coldout
            torch.cuda.empty_cache()
    from unifolm_wma.modules.attention import SpatialTransformer,CrossAttention
    with torch.device('meta'):
        tree=torch.nn.Module()
        tree.low=SpatialTransformer(320,5,64,depth=1)
        tree.wide=SpatialTransformer(640,10,64,depth=1)
        tree.outside=CrossAttention(320,heads=5,dim_head=64)
    for module in [tree.low.transformer_blocks[0].attn1,tree.wide.transformer_blocks[0].attn1,tree.outside]:configure(module)
    assert helper.install_world_qkv_packed_attention(tree)==2
    assert not hasattr(tree.outside,'_qkv_packed_attention_installed')
    try:helper.install_world_qkv_packed_attention(tree)
    except ValueError:pass
    else:raise AssertionError('duplicate installation')
    print('QKV_PACKED_ATTENTION_RUNTIME_CHECK passed=true fixtures=4 native_fallbacks=11 '
          'cold_capture=true graph_master_output_bias_refresh=true static_cache_same_storage=true '
          'factory_real_spatial_only=true outside_excluded=true duplicate_rejected=true',flush=True)


if __name__=='__main__':main()
