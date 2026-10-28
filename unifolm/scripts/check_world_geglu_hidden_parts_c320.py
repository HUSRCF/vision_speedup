"""Check exact block math, cache timing, graph refresh, and fallback dispatch."""
import json
import os
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.attention import BasicTransformerBlock,GEGLU
from unifolm_wma.utils import world_geglu_hidden_parts_c320 as helper
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight,half_compensated_linear
from unifolm_wma.utils.triton_zero_weight_low_geglu import residual_count
from unifolm_wma.utils.triton_world_narrow_geglu import primary_narrow_geglu
from unifolm_wma.utils.triton_branch_protected_geglu import branch_protected_geglu
from unifolm_wma.utils.triton_prepacked_geglu import prepacked_geglu


def exact(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape
    assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


class AffineAttention(torch.nn.Module):
    def forward(self,value,context=None,mask=None):
        result=value*.125+.01
        if context is not None:result=result+context.mean()*.00001
        return result


def configure(block):
    c=block.norm3.normalized_shape[0];ge=block.ff.net[0]
    block._inference_fused_residual_norm=True
    block._world_norm3_packed_geglu_installed=True
    block._world_ffn_residual_epilogue_installed=True
    block.ff.net[2]._tf32x3_installed=True
    if c==320:ge._world_bk64_c320_geglu_installed=True
    else:
        ge._prepacked_geglu_installed=True
        if c==640:ge._world_wide_gate_geglu_installed=True


def fixture(channels,bias):
    block=BasicTransformerBlock(channels,channels//64,64,checkpoint=False).cuda().eval()
    block.attn1=AffineAttention().eval();block.attn2=AffineAttention().eval()
    if not bias:
        block.ff.net[0].proj.register_parameter('bias',None)
        block.ff.net[2].register_parameter('bias',None)
    configure(block);ge=block.ff.net[0];p=ge.proj
    @torch.no_grad()
    def original_geglu(value):
        parts=pack_half_weight(p.weight);residual_count(parts,refresh=True)
        if channels==320:return primary_narrow_geglu(value,parts,p.bias,mode='gate',block_n=64)
        if channels==640:return branch_protected_geglu(value,parts,p.bias,mode='gate',block_n=64,group_m=32)
        return prepacked_geglu(value,parts,p.bias,block_m=64,block_n=32,group_m=32,num_warps=4,num_stages=2)
    ge.forward=original_geglu
    final=block.ff.net[2]
    @torch.no_grad()
    def original_final(value):
        return half_compensated_linear(value,final.weight,final.bias,
            block_m=64,block_n=64,block_k=64,group_m=32,weight_parts=pack_half_weight(final.weight))
    final.forward=original_final
    return block


def sentinel(value,context=None,mask=None):return value*.75+.25


@torch.no_grad()
def main():
    torch.manual_seed(832);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU='all',UNIFOLM_NORM_BUFFER_NOFILL='all',
                     UNIFOLM_HALF_PACK_NOFILL='1',UNIFOLM_GEGLU_OUTPUT_NOFILL='1')
    for channels,rows in [(320,40960)]:
        for bias in [False,True]:
            block=fixture(channels,bias);ge=block.ff.net[0];p=ge.proj
            p.weight.copy_(p.weight.half().float())
            keys=set(block.state_dict());masters=[t.data_ptr() for t in block.parameters()]
            original=block._forward;forward=helper.make_forward(block)
            value=torch.randn(16,rows//16,channels,device='cuda')*.3
            context=torch.randn(1,1,channels,device='cuda')*.1
            packs=[];pack=helper.pack_half_weight
            def recorded(*values,**options):
                result=pack(*values,**options);packs.append(tuple(x.data_ptr() for x in result));return result
            with patch.object(helper,'pack_half_weight',side_effect=recorded):
                assert not hasattr(ge,'_norm3_geglu_weight_cache') and not packs
                actual=forward(value,context=context);exact(actual,original(value,context=context));assert len(packs)==2
                cache=ge._norm3_geglu_weight_cache
                pointers=tuple(x.data_ptr() for x in cache[1]),cache[2].data_ptr()
                assert int(cache[2])==0
                final_pointers=tuple(t.data_ptr() for t in block.ff.net[2]._ffn_residual_weight_cache[1])
                for _ in range(3):exact(actual,forward(value,context=context))
                assert len(packs)==2
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):forward(value,context=context)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):captured=forward(value,context=context)
                for quantized in [False,True]:
                    value.add_(.0001);context.add_(.0002)
                    p.weight.add_(.00013)
                    if quantized:p.weight.copy_(p.weight.half().float())
                    if p.bias is not None:p.bias.add_(.0002)
                    block.norm3.weight.mul_(1.001);block.norm3.bias.add_(.0003)
                    block.ff.net[2].weight.add_(.000001)
                    if block.ff.net[2].bias is not None:block.ff.net[2].bias.add_(.0001)
                    eager=forward(value,context=context);cache=ge._norm3_geglu_weight_cache
                    assert pointers==(tuple(x.data_ptr() for x in cache[1]),cache[2].data_ptr())
                    assert (int(cache[2])==0)==quantized
                    assert final_pointers==tuple(t.data_ptr() for t in block.ff.net[2]._ffn_residual_weight_cache[1])
                    graph.replay();exact(captured,eager);exact(captured,original(value,context=context))
                assert len(packs)==6
            assert keys==set(block.state_dict()) and masters==[t.data_ptr() for t in block.parameters()]
            assert all(t.dtype==torch.float32 for t in block.parameters())
            # Separate dispatch-sentinel checks; actual attention and full-model
            # behavior are verified in the independent both-stage graph pair.
            block._forward=sentinel;fallback_forward=helper.make_forward(block)
            with patch.object(helper,'residual_norm_half_pack',side_effect=AssertionError('fallback entered fusion')):
                for target in [torch.backends.cudnn,torch.backends.cuda.matmul]:
                    target.allow_tf32=True;exact(fallback_forward(value),sentinel(value));target.allow_tf32=False
                block.train();exact(fallback_forward(value),sentinel(value));block.eval()
                with torch.enable_grad():exact(fallback_forward(value),sentinel(value))
                with torch.autocast('cuda',dtype=torch.float16):exact(fallback_forward(value),sentinel(value))
                for other in [value[:1],value.half(),value.transpose(1,2).contiguous().transpose(1,2),value.cpu()]:
                    left=fallback_forward(other);right=sentinel(other)
                    assert left.dtype==right.dtype and torch.equal(left,right)
                block.ff._inference_dtype=torch.float16;exact(fallback_forward(value),sentinel(value));block.ff._inference_dtype=torch.float32
                block.norm3.train();exact(fallback_forward(value),sentinel(value));block.norm3.eval()
            cold=fixture(channels,bias);cold._forward=sentinel;cold_forward=helper.make_forward(cold)
            with patch.object(helper,'pack_half_weight',side_effect=AssertionError('cold graph initialized cache')):
                coldgraph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(coldgraph):coldout=cold_forward(value)
                coldgraph.replay();exact(coldout,sentinel(value))
            assert not hasattr(cold.ff.net[0],'_norm3_geglu_weight_cache')
            assert not hasattr(cold.ff.net[2],'_ffn_residual_weight_cache')
            from unifolm_wma.utils import world_norm3_packed_geglu as norm_helper
            norm_helper.make_forward(cold)(value)
            assert hasattr(cold.ff.net[0],'_norm3_geglu_weight_cache')
            assert not hasattr(cold.ff.net[2],'_ffn_residual_weight_cache')
            with patch.object(helper,'pack_half_weight',side_effect=AssertionError('cold final cache packed in capture')):
                finalcoldgraph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(finalcoldgraph):finalcoldout=cold_forward(value)
                finalcoldgraph.replay();exact(finalcoldout,sentinel(value))
            assert not hasattr(cold.ff.net[2],'_ffn_residual_weight_cache')
            print('WORLD_GEGLU_HIDDEN_PARTS_C320_RUNTIME '+json.dumps(dict(channels=channels,rows=rows,bias=bias,
                real_block_with_affine_attention_reference_signedzero=True,
                both_caches_first_timed_warm_once=True,both_caches_version_refresh_same_storage=True,
                graph_input_context_norm3_master_bias_geglu_master_bias_final_master_bias=True,
                zero_nonzero_zero_device_predicate=True,fp32_master_storage_state_preserved=True,
                native_fallbacks=11,cold_capture_fallback=True,final_only_cold_capture_fallback=True,fallback_fixture_is_dispatch_sentinel=True)),flush=True)
            del block,ge,p,original,forward,value,context,actual,eager,captured,graph,cache,cold,cold_forward,coldgraph,coldout,fallback_forward
            torch.cuda.empty_cache()
    with torch.device('meta'):
        tree=torch.nn.Module()
        tree.low=BasicTransformerBlock(320,5,64);tree.mid=BasicTransformerBlock(640,10,64)
        tree.wide=BasicTransformerBlock(1280,20,64);tree.initial=BasicTransformerBlock(512,8,64)
        tree.outside=GEGLU(640,2560)
    for block in [tree.low,tree.mid,tree.wide]:configure(block)
    assert helper.install_world_geglu_hidden_parts_c320(tree)==1
    assert not hasattr(tree.initial,'_world_geglu_hidden_parts_c320_installed')
    assert not hasattr(tree.mid,'_world_geglu_hidden_parts_c320_installed')
    assert not hasattr(tree.wide,'_world_geglu_hidden_parts_c320_installed')
    assert not hasattr(tree.outside,'_world_geglu_hidden_parts_c320_installed')
    try:helper.install_world_geglu_hidden_parts_c320(tree)
    except ValueError:pass
    else:raise AssertionError('duplicate installation')
    print('WORLD_GEGLU_HIDDEN_PARTS_C320_RUNTIME_CHECK passed=true fixtures=2 fallback_dispatches=11 '
          'cold_capture=true cache_first_timed_warm_version_refresh=true original_state_masters=true '
          'real_block_math_affine_attention=true factory_real_blocks=true initial512_outside_excluded=true duplicate_rejected=true',flush=True)


if __name__=='__main__':main()
