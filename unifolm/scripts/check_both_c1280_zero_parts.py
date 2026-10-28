"""Both current C1280 stage wrappers share versioned GE/final storage."""
import gc
import json
import os
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.utils import decision_primary_hidden_parts as decision
from unifolm_wma.utils import world_geglu_hidden_parts_c1280_zero as world
from check_decision_primary_hidden_parts import fixture, golden, exact


@torch.no_grad()
def main():
    torch.manual_seed(1401)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU='all', UNIFOLM_NORM_BUFFER_NOFILL='all',
        UNIFOLM_GEMM_OUTPUT_NOFILL='all', UNIFOLM_HALF_PACK_NOFILL='1', UNIFOLM_GEGLU_OUTPUT_NOFILL='1')
    fixtures = 0
    for rows in [2560,640]:
        for bias in [False,True]:
            for first in ['decision','world']:
                block = fixture(bias)
                block._forward = decision.make_forward(block)
                block._decision_primary_hidden_parts_installed = True
                forward = world.make_forward(block)
                ge, final = block.ff.net[0], block.ff.net[2]
                proj = ge.proj
                proj.weight.copy_(proj.weight.half().float())
                final.weight.copy_(final.weight.half().float())
                masters = [p.data_ptr() for p in block.parameters()]
                keys = set(block.state_dict())
                value = torch.randn(16,rows//16,1280,device='cuda')*.3
                context = torch.randn(1,1,1280,device='cuda')*.1
                packs = []
                pack = decision.pack_half_weight
                def recorder(stage):
                    def recorded(*args,**kwargs):
                        result = pack(*args,**kwargs)
                        packs.append(stage)
                        return result
                    return recorded
                def call(stage):
                    torch.backends.cudnn.allow_tf32 = stage=='decision'
                    return forward(value,context=context)
                def pointers():
                    cache = ge._norm3_geglu_weight_cache
                    return (tuple(t.data_ptr() for t in cache[1]),cache[2].data_ptr(),
                            tuple(t.data_ptr() for t in final._decision_ffn_residual_weight_cache[1]),
                            final._decision_ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count[1].data_ptr())
                with patch.object(decision,'pack_half_weight',side_effect=recorder('decision')), \
                     patch.object(world,'pack_half_weight',side_effect=recorder('world')):
                    for stage in [first,'world' if first=='decision' else 'decision']:
                        exact(call(stage),golden(block,value,context))
                    assert packs == [first,first], packs
                    storage = pointers()
                    assert int(ge._norm3_geglu_weight_cache[2]) == 0
                    assert int(final._decision_ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count[1])==0
                    assert final._world_ffn1280_low_count_key==final._decision_ffn_residual_weight_cache[0]
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for stage in ['world','decision']:
                            for _ in range(3):call(stage)
                    torch.cuda.current_stream().wait_stream(stream)
                    graphs, outputs = {}, {}
                    for stage in ['world','decision']:
                        torch.backends.cudnn.allow_tf32 = stage=='decision'
                        graphs[stage] = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graphs[stage],stream=stream):
                            outputs[stage] = forward(value,context=context)
                    assert len(packs) == 2
                    # One stage alone refreshes both shared weight buffers;
                    # then both previously captured graphs must see the update.
                    for refresh_stage, quantized in [('decision',False),('world',True),('world',False),('decision',True)]:
                        value.add_(.0001);context.add_(.0002)
                        proj.weight.add_(.00013);final.weight.add_(.00017)
                        if quantized:
                            proj.weight.copy_(proj.weight.half().float())
                            final.weight.copy_(final.weight.half().float())
                        if proj.bias is not None:proj.bias.add_(.0002);final.bias.add_(.0003)
                        for name in ['norm1','norm2','norm3']:
                            norm = getattr(block,name)
                            norm.weight.mul_(1.001);norm.bias.add_(.0003)
                        exact(call(refresh_stage),golden(block,value,context))
                        assert pointers() == storage
                        assert (int(ge._norm3_geglu_weight_cache[2])==0) == quantized
                        assert (int(final._decision_ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count[1])==0)==quantized
                        assert final._world_ffn1280_low_count_key==final._decision_ffn_residual_weight_cache[0]
                        for stage in ['world','decision']:
                            torch.backends.cudnn.allow_tf32 = stage=='decision'
                            graphs[stage].replay()
                            exact(outputs[stage],golden(block,value,context))
                    assert len(packs) == 10
                # Missing count or stale masters during capture must delegate;
                # never allocate or silently freeze a stale static predicate.
                for cold_kind in ['both','count_missing','GE_stale','final_stale']:
                    cold=fixture(bias);cold._decision_primary_hidden_parts_installed=True
                    cold._forward=lambda x,context=None,mask=None:x*.75+.25
                    cold_forward=world.make_forward(cold)
                    torch.backends.cudnn.allow_tf32=False
                    if cold_kind!='both':
                        cold_forward(value,context=context)
                        if cold_kind=='count_missing':
                            del cold.ff.net[2]._decision_ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count
                        elif cold_kind=='GE_stale':cold.ff.net[0].proj.weight.add_(.00013)
                        else:cold.ff.net[2].weight.add_(.00017)
                    with patch.object(world,'pack_half_weight',side_effect=AssertionError('cold capture packed weights')):
                        cg=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(cg):co=cold_forward(value,context=context)
                        cg.replay();exact(co,value*.75+.25)
                    cg.reset()
                    del cold,cold_forward,cg,co
                assert masters == [p.data_ptr() for p in block.parameters()]
                assert keys == set(block.state_dict())
                assert all(p.dtype==torch.float32 for p in block.parameters())
                print('BOTH_C1280_HIDDEN_RUNTIME '+json.dumps(dict(rows=rows,bias=bias,first_stage=first,
                    independent_original_math_reference_signedzero=True,first_timed_pack_once_per_weight=True,
                    both_stage_graph_cross_refresh_both_weights=True,zero_nonzero_zero_device_predicate=True,
                    shared_GE_count_final_storage_unchanged=True,FFN_count_refreshes_by_original_master_version=True,
                    FFN_count_cross_stage_same_pointer=True,cold_or_stale_capture_fallback4=True,
                    FP32_masters_state_preserved=True)),flush=True)
                fixtures += 1
                for graph in graphs.values():graph.reset()
                del block,forward,ge,final,proj,value,context,graphs,outputs
                gc.collect();torch.cuda.empty_cache()
    assert fixtures == 8
    print('BOTH_C1280_HIDDEN_RUNTIME_CHECK passed=true fixtures=8 both_warm_orders=true both_stage_cross_refresh=true signedzero=true',flush=True)


if __name__ == '__main__':
    main()
