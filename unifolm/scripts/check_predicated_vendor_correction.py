"""Real C640 blocks, both stage graphs and shared cache refresh guards."""
import gc
import json
import os
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.attention import BasicTransformerBlock,GEGLU
from unifolm_wma.utils import decision_chunked_hidden_parts as helper
from unifolm_wma.utils import world_geglu_hidden_parts as world_helper
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_primary_geglu import primary_geglu
from unifolm_wma.utils.triton_branch_protected_geglu import branch_protected_geglu
from unifolm_wma.utils.cublas_chunked_compensated_linear import chunked_compensated_linear
from unifolm_wma.utils.ffn640_residual_epilogue import ffn640_residual_epilogue
from unifolm_wma.utils.vendor_correction_count import get_count,refresh_count
from predicated_vendor_correction_hooks import installed
from unifolm_wma.utils.decision_chunked_hidden_pipeline import chunked_hidden_residual
from unifolm_wma.utils.triton_residual_layer_norm import fused_residual_layer_norm


def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


class AffineAttention(torch.nn.Module):
    def forward(self,value,context=None,mask=None):
        output=value*.125+.01
        if context is not None:output=output+context.mean()*.00001
        return output


def configure(block):
    for key in ['_inference_fused_residual_norm','_world_norm3_packed_geglu_installed',
                '_world_ffn640_residual_epilogue_installed','_world_geglu_hidden_parts_installed']:
        setattr(block,key,True)
    for key in ['_prepacked_geglu_installed','_decision_primary_geglu_installed','_world_wide_gate_geglu_installed']:
        setattr(block.ff.net[0],key,True)
    for key in ['_tf32x3_installed','_decision_vendor_ffn_installed','_world_two_product_roles_installed']:
        setattr(block.ff.net[2],key,True)


def fixture(bias):
    block=BasicTransformerBlock(640,10,64,checkpoint=False).cuda().eval()
    block.attn1=AffineAttention().eval();block.attn2=AffineAttention().eval()
    if not bias:
        block.ff.net[0].proj.register_parameter('bias',None);block.ff.net[2].register_parameter('bias',None)
    configure(block);ge=block.ff.net[0];proj=ge.proj;final=block.ff.net[2]
    def original_geglu(value):
        parts=pack_half_weight(proj.weight)
        if torch.backends.cudnn.allow_tf32:return primary_geglu(value,parts,proj.bias,mode='both')
        return branch_protected_geglu(value,parts,proj.bias,mode='gate',block_n=64,group_m=32)
    def original_final(value):
        return chunked_compensated_linear(value,pack_half_weight(final.weight),final.bias,chunk_k=512)
    ge.forward=original_geglu;final.forward=original_final
    block._forward=world_helper.make_forward(block)
    return block


def golden(block,value,context):
    attended=block.attn1(block.norm1(value),context=context if block.disable_self_attn else None)
    residual,normalized=fused_residual_layer_norm(attended,value,block.norm2)
    attended=block.attn2(normalized,context=context)
    residual,normalized=fused_residual_layer_norm(attended,residual,block.norm3)
    ge=block.ff.net[0].proj;final=block.ff.net[2];gp=pack_half_weight(ge.weight);fp=pack_half_weight(final.weight)
    if torch.backends.cudnn.allow_tf32:
        hidden=primary_geglu(normalized,gp,ge.bias,mode='both')
        result=chunked_compensated_linear(hidden,fp,final.bias,chunk_k=512)
        assert result is not None
        return result+residual
    hidden=branch_protected_geglu(normalized,gp,ge.bias,mode='gate',block_n=64,group_m=32)
    return ffn640_residual_epilogue(hidden,fp,residual,final.bias)


def sentinel(value,context=None,mask=None):return value*.75+.25


@torch.no_grad()
def real_guards():
    torch.manual_seed(1141);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU='all',UNIFOLM_NORM_BUFFER_NOFILL='all',
        UNIFOLM_GEMM_OUTPUT_NOFILL='all',UNIFOLM_HALF_PACK_NOFILL='1',UNIFOLM_GEGLU_OUTPUT_NOFILL='1')
    os.environ['UNIFOLM_DECISION_VENDOR_CORRECTION']='1'
    for bias in [False,True]:
        for first in ['world','decision']:
            block=fixture(bias);ge=block.ff.net[0];proj=ge.proj;final=block.ff.net[2]
            proj.weight.copy_(proj.weight.half().float());final.weight.copy_(final.weight.half().float())
            keys=set(block.state_dict());masters=[p.data_ptr() for p in block.parameters()]
            forward=helper.make_forward(block)
            value=torch.randn(16,640,640,device='cuda')*.3;context=torch.randn(1,1,640,device='cuda')*.1
            packs=[];pack=helper.pack_half_weight
            def recorder(stage):
                def recorded(*args,**kwargs):
                    result=pack(*args,**kwargs);packs.append(stage);return result
                return recorded
            def call(stage):
                torch.backends.cudnn.allow_tf32=stage=='decision'
                return forward(value,context=context)
            def pointers():
                cache=ge._norm3_geglu_weight_cache
                return tuple(p.data_ptr() for p in cache[1]),cache[2].data_ptr(),tuple(p.data_ptr() for p in final._ffn640_residual_weight_cache[1]),get_count(final._ffn640_residual_weight_cache[1]).data_ptr()
            with patch.object(helper,'pack_half_weight',side_effect=recorder('decision')),patch.object(world_helper,'pack_half_weight',side_effect=recorder('world')):
                assert not hasattr(ge,'_norm3_geglu_weight_cache') and not hasattr(final,'_ffn640_residual_weight_cache')
                for stage in [first,'decision' if first=='world' else 'world']:exact(call(stage),golden(block,value,context))
                assert packs==[first,first];storage=pointers();assert int(ge._norm3_geglu_weight_cache[2])==0
                assert int(get_count(final._ffn640_residual_weight_cache[1]))==0
                for stage in ['world','decision']:
                    for _ in range(2):exact(call(stage),golden(block,value,context))
                assert len(packs)==2
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for stage in ['world','decision']:
                        for _ in range(3):call(stage)
                torch.cuda.current_stream().wait_stream(stream);graphs={};captured={}
                for stage in ['world','decision']:
                    torch.backends.cudnn.allow_tf32=stage=='decision';graphs[stage]=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graphs[stage],stream=stream):captured[stage]=forward(value,context=context)
                assert len(packs)==2
                for quantized in [False,True]:
                    value.add_(.0001);context.add_(.0002);proj.weight.add_(.00013);final.weight.add_(.00013)
                    if quantized:proj.weight.copy_(proj.weight.half().float());final.weight.copy_(final.weight.half().float())
                    if proj.bias is not None:proj.bias.add_(.0002);final.bias.add_(.0001)
                    for name in ['norm1','norm2','norm3']:
                        norm=getattr(block,name);norm.weight.mul_(1.001);norm.bias.add_(.0003)
                    # Refresh from one stage only; both captured graphs must see it.
                    refreshed='decision' if not quantized else 'world'
                    exact(call(refreshed),golden(block,value,context))
                    assert pointers()==storage and (int(ge._norm3_geglu_weight_cache[2])==0)==quantized
                    assert (int(get_count(final._ffn640_residual_weight_cache[1]))==0)==quantized
                    for stage in ['world','decision']:
                        torch.backends.cudnn.allow_tf32=stage=='decision';graphs[stage].replay();exact(captured[stage],golden(block,value,context))
                assert len(packs)==6
                proj.weight.add_(.000043);call('decision');assert len(packs)==7
                torch.backends.cudnn.allow_tf32=False;graphs['world'].replay();exact(captured['world'],golden(block,value,context))
                assert int(ge._norm3_geglu_weight_cache[2])!=0
                proj.weight.copy_(proj.weight.half().float());call('world');assert len(packs)==8
                torch.backends.cudnn.allow_tf32=True;graphs['decision'].replay();exact(captured['decision'],golden(block,value,context))
                assert int(ge._norm3_geglu_weight_cache[2])==0
                for stage in ['decision','world']:
                    final.weight.add_(.000043);call(stage)
                    other='world' if stage=='decision' else 'decision';torch.backends.cudnn.allow_tf32=other=='decision'
                    graphs[other].replay();exact(captured[other],golden(block,value,context))
                assert len(packs)==10 and pointers()==storage
                assert int(get_count(final._ffn640_residual_weight_cache[1]))!=0
            assert keys==set(block.state_dict()) and masters==[p.data_ptr() for p in block.parameters()] and all(p.dtype==torch.float32 for p in block.parameters())
            torch.backends.cudnn.allow_tf32=True;block._forward=sentinel;fallback=helper.make_forward(block)
            with patch.object(helper,'residual_norm_half_pack',side_effect=AssertionError('guard entered fusion')):
                torch.backends.cudnn.allow_tf32=False;exact(fallback(value),sentinel(value));torch.backends.cudnn.allow_tf32=True
                torch.backends.cuda.matmul.allow_tf32=True;exact(fallback(value),sentinel(value));torch.backends.cuda.matmul.allow_tf32=False
                block.train();exact(fallback(value),sentinel(value));block.eval()
                with torch.enable_grad():exact(fallback(value),sentinel(value))
                with torch.autocast('cuda',dtype=torch.float16):exact(fallback(value),sentinel(value))
                for other in [value[:1],value.half(),value.transpose(1,2).contiguous().transpose(1,2),value.cpu()]:
                    a=fallback(other);b=sentinel(other);assert a.dtype==b.dtype and torch.equal(a,b)
                block.ff._inference_dtype=torch.float16;exact(fallback(value),sentinel(value));block.ff._inference_dtype=torch.float32
                block.ff.net[2]._decision_primary_ffn_installed=True;exact(fallback(value),sentinel(value));block.ff.net[2]._decision_primary_ffn_installed=False
            for kind in ['both','only_GE_warm','only_final_warm','GE_stale','final_stale']:
                cold=fixture(bias);cold._forward=sentinel;cold_forward=helper.make_forward(cold)
                if kind!='both':
                    cold_forward(value)
                    if kind=='only_GE_warm':del cold.ff.net[2]._ffn640_residual_weight_cache
                    if kind=='only_final_warm':del cold.ff.net[0]._norm3_geglu_weight_cache
                    if kind.endswith('_stale'):
                        p=cold.ff.net[0].proj if kind=='GE_stale' else cold.ff.net[2];p.weight.add_(.00013)
                with patch.object(helper,'pack_half_weight',side_effect=AssertionError('capture packed weights')):
                    cg=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(cg):co=cold_forward(value)
                    cg.replay();exact(co,sentinel(value))
                del cold,cold_forward,cg,co
            with patch.object(helper,'_READY',set()):
                readygraph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(readygraph):readyout=fallback(value)
                readygraph.replay();exact(readyout,sentinel(value))
            print('DECISION_CHUNKED_HIDDEN_PARTS_RUNTIME '+json.dumps(dict(rows=10240,bias=bias,first_stage=first,
                real_block_independent_both_stage_reference_signedzero=True,first_timed_shared_caches_both_warm_orders=True,
                graph_inputs_context_all_norms_GE_final_masters_bias=True,shared_GE_and_final_refresh_either_stage_to_other_graph=True,
                version_refresh_same_storage=True,zero_nonzero_zero_GE_predicate=True,fp32_master_state_preserved=True,
                cold_partial_and_each_stale_capture_fallback=True,missing_cublas_ready_fallback=True,native_fallbacks=11)),flush=True)
            del block,ge,proj,final,forward,value,context,graphs,captured,fallback,readygraph,readyout
            gc.collect();torch.cuda.empty_cache()
    with torch.device('meta'):
        tree=torch.nn.Module();tree.mid=torch.nn.ModuleList([BasicTransformerBlock(640,10,64) for _ in range(10)])
        tree.low=BasicTransformerBlock(320,5,64);tree.wide=BasicTransformerBlock(1280,20,64);tree.initial=BasicTransformerBlock(512,8,64);tree.outside=GEGLU(640,2560)
    for block in tree.mid:configure(block)
    assert helper.install_decision_chunked_hidden_parts(tree)==10
    assert all(not getattr(m,'_decision_chunked_hidden_parts_installed',False) for m in [tree.low,tree.wide,tree.initial,tree.outside])
    try:helper.install_decision_chunked_hidden_parts(tree)
    except ValueError:pass
    else:raise AssertionError('duplicate installation accepted')
    print('DECISION_CHUNKED_HIDDEN_PARTS_RUNTIME_CHECK passed=true fixtures=4 both_stage_shared_GE_and_final_cross_refresh=true fallback11_coldpartialstale5_ready=true factory10=true duplicate_rejected=true')


@torch.no_grad()
def main():
    with installed() as launches:
        real_guards()
        os.environ['UNIFOLM_DECISION_VENDOR_CORRECTION']='1'
        torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
        raw=torch.randn(10240,2560,device='cuda')*.3
        inputs=pack_half_weight(raw,weight_scale=1.)
        parts=pack_half_weight(torch.randn(640,2560,device='cuda')*.02)
        residual=torch.randn(10240,640,device='cuda')*.2
        bias=torch.randn(640,device='cuda')*.02
        assert get_count(parts) is not None and int(get_count(parts))!=0
        for policy in ['selected','disabled','world','matmul_tf32','other_rows','other_chunk',
                       'missing_count','stale_count','invalid_count_dtype']:
            torch.backends.cudnn.allow_tf32=policy!='world'
            torch.backends.cuda.matmul.allow_tf32=policy=='matmul_tf32'
            xp=tuple(p[:2560] for p in inputs) if policy=='other_rows' else inputs
            rp=residual[:2560] if policy=='other_rows' else residual
            chunk=256 if policy=='other_chunk' else 512
            saved=getattr(parts[0],'_decision_vendor_correction_count')
            if policy=='missing_count':del parts[0]._decision_vendor_correction_count
            if policy=='stale_count':parts[0].add_(0)
            if policy=='invalid_count_dtype':
                parts[0]._decision_vendor_correction_count=(saved[0],torch.zeros((),device='cuda'))
            os.environ['UNIFOLM_DECISION_VENDOR_CORRECTION']='0'
            expected=chunked_hidden_residual(xp,parts,rp,bias,chunk_k=chunk)
            os.environ['UNIFOLM_DECISION_VENDOR_CORRECTION']='0' if policy=='disabled' else '1'
            before=len(launches);actual=chunked_hidden_residual(xp,parts,rp,bias,chunk_k=chunk)
            exact(actual,expected)
            assert len(launches)-before==(1 if policy=='selected' else 0)
            parts[0]._decision_vendor_correction_count=saved
            if policy=='stale_count':refresh_count(parts)
        torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
        os.environ['UNIFOLM_DECISION_VENDOR_CORRECTION']='1'
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):chunked_hidden_residual(inputs,parts,residual,bias)
        torch.cuda.current_stream().wait_stream(stream)
        saved=parts[0]._decision_vendor_correction_count;del parts[0]._decision_vendor_correction_count
        before=len(launches);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=chunked_hidden_residual(inputs,parts,residual,bias)
        assert captured is not None and len(launches)==before
        graph.replay();exact(captured,chunked_hidden_residual(inputs,parts,residual,bias))
        parts[0]._decision_vendor_correction_count=saved
        os.environ['UNIFOLM_DECISION_VENDOR_CORRECTION']='invalid'
        try:chunked_hidden_residual(inputs,parts,residual,bias)
        except ValueError as e:assert 'vendor correction' in str(e)
        else:raise AssertionError('Invalidcorrectionmodeaccepted')
        os.environ['UNIFOLM_DECISION_VENDOR_CORRECTION']='1'
    print('PREDICATED_VENDOR_CORRECTION_RUNTIME '+json.dumps(dict(passed=True,
        real_block_fixtures=4,policy_fixtures=10,final_predicate_zero_nonzero_zero=True,
        shared_final_master_refresh_either_stage_to_decision_graph=True,
        count_same_storage_cross_stage=True,original_real_block_guards_retained=True,
        old_vendor_reference_signedzero=True,cold_stale_missing_dtype_shape_backend_mode_fallbacks=True,
        cold_count_capture_original_vendor=True,invalid_mode_rejected=True,
        actual_kernel_launches_observed=len(launches))),flush=True)
    print('PREDICATED_VENDOR_CORRECTION_RUNTIME_CHECK passed=true fixtures=4 policies=10',flush=True)


if __name__=='__main__':
    main()
