"""Real-block math and lifecycle guards; actual attention checked separately."""
import json
import os
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.attention import BasicTransformerBlock,GEGLU
from unifolm_wma.utils import decision_ffn_residual_epilogue as helper
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.decision_primary_ffn import primary_linear


def exact(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape
    assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


class AffineAttention(torch.nn.Module):
    def forward(self,value,context=None,mask=None):
        result=value*.125+.01
        if context is not None:result=result+context.mean()*.00001
        return result


def configure(block):
    block._inference_fused_residual_norm=True
    final=block.ff.net[2]
    for key in ['_tf32x3_installed','_decision_vendor_ffn_installed','_decision_primary_ffn_installed']:
        setattr(final,key,True)


def fixture(bias):
    block=BasicTransformerBlock(1280,20,64,checkpoint=False).cuda().eval()
    block.attn1=AffineAttention().eval();block.attn2=AffineAttention().eval()
    if not bias:
        block.ff.net[0].proj.register_parameter('bias',None)
        block.ff.net[2].register_parameter('bias',None)
    configure(block);final=block.ff.net[2]
    @torch.no_grad()
    def original_final(value):return primary_linear(value,pack_half_weight(final.weight)[0],final.bias)
    final.forward=original_final
    return block


def sentinel(value,context=None,mask=None):return value*.75+.25


@torch.no_grad()
def main():
    torch.manual_seed(910);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(UNIFOLM_NORM_BUFFER_NOFILL='all',UNIFOLM_HALF_PACK_NOFILL='1')
    for rows in [2560,640]:
        for bias in [False,True]:
            block=fixture(bias);final=block.ff.net[2];ge=block.ff.net[0].proj
            final.weight.copy_(final.weight.half().float())
            keys=set(block.state_dict());masters=[t.data_ptr() for t in block.parameters()]
            original=block._forward;forward=helper.make_forward(block)
            value=torch.randn(16,rows//16,1280,device='cuda')*.3
            context=torch.randn(1,1,1280,device='cuda')*.1
            packs=[];pack=helper.pack_half_weight
            def recorded(*values,**options):
                result=pack(*values,**options);packs.append(tuple(x.data_ptr() for x in result));return result
            with patch.object(helper,'pack_half_weight',side_effect=recorded):
                assert not hasattr(final,'_decision_ffn_residual_weight_cache') and not packs
                actual=forward(value,context=context);exact(actual,original(value,context=context));assert len(packs)==1
                pointers=tuple(t.data_ptr() for t in final._decision_ffn_residual_weight_cache[1])
                for _ in range(3):exact(actual,forward(value,context=context))
                assert len(packs)==1
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):forward(value,context=context)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):captured=forward(value,context=context)
                for quantized in [False,True]:
                    value.add_(.0001);context.add_(.0002);ge.weight.add_(.00013)
                    if ge.bias is not None:ge.bias.add_(.0002)
                    for name in ['norm1','norm2','norm3']:
                        norm=getattr(block,name);norm.weight.mul_(1.001);norm.bias.add_(.0003)
                    final.weight.add_(.00013)
                    if quantized:final.weight.copy_(final.weight.half().float())
                    if final.bias is not None:final.bias.add_(.0001)
                    eager=forward(value,context=context)
                    assert pointers==tuple(t.data_ptr() for t in final._decision_ffn_residual_weight_cache[1])
                    graph.replay();exact(captured,eager);exact(captured,original(value,context=context))
                assert len(packs)==3
            assert keys==set(block.state_dict()) and masters==[t.data_ptr() for t in block.parameters()]
            assert all(t.dtype==torch.float32 for t in block.parameters())
            block._forward=sentinel;fallback=helper.make_forward(block)
            with patch.object(helper,'primary_linear_residual',side_effect=AssertionError('fallback entered fusion')):
                torch.backends.cudnn.allow_tf32=False;exact(fallback(value),sentinel(value));torch.backends.cudnn.allow_tf32=True
                torch.backends.cuda.matmul.allow_tf32=True;exact(fallback(value),sentinel(value));torch.backends.cuda.matmul.allow_tf32=False
                block.train();exact(fallback(value),sentinel(value));block.eval()
                with torch.enable_grad():exact(fallback(value),sentinel(value))
                with torch.autocast('cuda',dtype=torch.float16):exact(fallback(value),sentinel(value))
                for other in [value[:1],value.half(),value.transpose(1,2).contiguous().transpose(1,2),value.cpu()]:
                    left=fallback(other);right=sentinel(other)
                    assert left.dtype==right.dtype and torch.equal(left,right)
                block.ff._inference_dtype=torch.float16;exact(fallback(value),sentinel(value));block.ff._inference_dtype=torch.float32
                block.norm3.train();exact(fallback(value),sentinel(value));block.norm3.eval()
            cold=fixture(bias);cold._forward=sentinel;cold_forward=helper.make_forward(cold)
            with patch.object(helper,'pack_half_weight',side_effect=AssertionError('cold capture packed weights')):
                coldgraph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(coldgraph):coldout=cold_forward(value)
                coldgraph.replay();exact(coldout,sentinel(value))
            assert not hasattr(cold.ff.net[2],'_decision_ffn_residual_weight_cache')
            with patch.object(helper,'_READY',set()):
                readygraph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(readygraph):readyout=fallback(value)
                readygraph.replay();exact(readyout,sentinel(value))
            print('DECISION_FFN_RESIDUAL_EPILOGUE_RUNTIME '+json.dumps(dict(rows=rows,bias=bias,
                real_block_affine_attention_primary_projection_reference_signedzero=True,
                cache_first_timed_warm_once=True,cache_version_refresh_same_storage=True,
                graph_input_context_all_norms_geglu_master_bias_final_master_bias=True,
                fp32_master_storage_state_preserved=True,native_fallbacks=11,
                cold_cache_capture_fallback=True,cold_cublas_stream_capture_fallback=True,
                fallback_fixture_is_dispatch_sentinel=True)),flush=True)
            del block,final,ge,original,forward,value,context,actual,eager,captured,graph,cold,cold_forward,coldgraph,coldout,fallback,readygraph,readyout
            torch.cuda.empty_cache()
    with torch.device('meta'):
        tree=torch.nn.Module()
        tree.wide=BasicTransformerBlock(1280,20,64);tree.wide2=BasicTransformerBlock(1280,20,64)
        tree.low=BasicTransformerBlock(320,5,64);tree.mid=BasicTransformerBlock(640,10,64)
        tree.initial=BasicTransformerBlock(512,8,64);tree.outside=GEGLU(1280,5120)
    for block in [tree.wide,tree.wide2]:configure(block)
    assert helper.install_decision_ffn_residual_epilogue(tree)==2
    assert all(not getattr(m,'_decision_ffn_residual_epilogue_installed',False)
               for m in [tree.low,tree.mid,tree.initial,tree.outside])
    try:helper.install_decision_ffn_residual_epilogue(tree)
    except ValueError:pass
    else:raise AssertionError('duplicate installation')
    print('DECISION_FFN_RESIDUAL_EPILOGUE_RUNTIME_CHECK passed=true fixtures=4 real_block_math=true '
          'fallbacks=11 cold_cache_and_cublas_stream=true factory=true duplicate_rejected=true',flush=True)


if __name__=='__main__':main()
