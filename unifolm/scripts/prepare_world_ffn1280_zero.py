"""New outer world wrapper owns count refresh for both existing stage caches."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'
text=(utils/'world_geglu_hidden_parts_c1280.py').read_text()
def change(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
change('from .geglu_hidden_parts_c1280_pipeline import geglu_hidden_parts_c1280,packed_hidden_ffn1280_residual',
'''from .geglu_hidden_parts_c1280_pipeline import geglu_hidden_parts_c1280
from .ffn1280_zero_residual import shared_final_low_count,zero_ffn1280_hidden_residual''')
change("        and getattr(block,'_decision_ffn_residual_epilogue_installed',False))",
"        and getattr(block,'_decision_ffn_residual_epilogue_installed',False)\n        and getattr(block,'_decision_primary_hidden_parts_installed',False))")
change('def make_forward(block):', '''def final_count(final,cache):
    if cache is None:return None
    w=final.weight
    try:key=(w.data_ptr(),w._version,tuple(w.shape),w.dtype,w.device)
    except RuntimeError:return None
    if cache[0]!=key:return None
    previous=getattr(final,'_world_ffn1280_low_count_key',None)
    # Triton out= writes do not increment parts._version. The original master
    # key must force count refresh after either stage repacks the shared parts.
    count=shared_final_low_count(cache[1],refresh=(previous!=key))
    if count is not None:final._world_ffn1280_low_count_key=key
    return count


def decision_active(block,value):
    return (compatible(block) and not any(m.training for m in block.modules())
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
        and torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        and value.is_cuda and value.dtype==torch.float32 and value.ndim==3
        and value.is_contiguous() and value.shape[-1]==1280
        and value.numel()//1280 in {2560,640})


def make_forward(block):''')
change('    def forward(value,context=None,mask=None):\n        if not compatible(block)',
'''    def forward(value,context=None,mask=None):
        if decision_active(block,value):
            result=original(value,context=context,mask=mask)
            final=block.ff.net[2]
            final_count(final,getattr(final,'_decision_ffn_residual_weight_cache',None))
            return result
        if not compatible(block)''')
change('        attended=block.attn1(block.norm1(value),',
'''        count=final_count(final,final_cache)
        if count is None:return original(value,context=context,mask=mask)
        attended=block.attn1(block.norm1(value),''')
change('result=packed_hidden_ffn1280_residual(hidden_parts,final_cache[1],residual.view(-1,c),final.bias)',
       'result=zero_ffn1280_hidden_residual(hidden_parts,final_cache[1],count,residual.view(-1,c),final.bias)')
text=text.replace('_world_geglu_hidden_parts_c1280_calls','_world_geglu_hidden_parts_c1280_zero_calls')
text=text.replace('_world_geglu_hidden_parts_c1280_installed','_world_geglu_hidden_parts_c1280_zero_installed')
text=text.replace('install_world_geglu_hidden_parts_c1280(', 'install_world_geglu_hidden_parts_c1280_zero(')
ast.parse(text);(utils/'world_geglu_hidden_parts_c1280_zero.py').write_text(text)

guard=(root/'scripts/check_both_c1280_hidden_parts.py').read_text()
def guard_change(old,new):
    global guard
    assert guard.count(old)==1,old
    guard=guard.replace(old,new)
guard_change('world_geglu_hidden_parts_c1280 as world','world_geglu_hidden_parts_c1280_zero as world')
guard_change('tuple(t.data_ptr() for t in final._decision_ffn_residual_weight_cache[1]))',
'''tuple(t.data_ptr() for t in final._decision_ffn_residual_weight_cache[1]),
                            final._decision_ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count[1].data_ptr())''')
guard_change('                    assert int(ge._norm3_geglu_weight_cache[2]) == 0',
'''                    assert int(ge._norm3_geglu_weight_cache[2]) == 0
                    assert int(final._decision_ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count[1])==0
                    assert final._world_ffn1280_low_count_key==final._decision_ffn_residual_weight_cache[0]''')
guard_change('                        assert (int(ge._norm3_geglu_weight_cache[2])==0) == quantized',
'''                        assert (int(ge._norm3_geglu_weight_cache[2])==0) == quantized
                        assert (int(final._decision_ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count[1])==0)==quantized
                        assert final._world_ffn1280_low_count_key==final._decision_ffn_residual_weight_cache[0]''')
guard_change('                assert masters == [p.data_ptr() for p in block.parameters()]',
'''                # Missing count or stale masters during capture must delegate;
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
                assert masters == [p.data_ptr() for p in block.parameters()]''')
guard_change('shared_GE_count_final_storage_unchanged=True,FP32_masters_state_preserved=True',
'''shared_GE_count_final_storage_unchanged=True,FFN_count_refreshes_by_original_master_version=True,
                    FFN_count_cross_stage_same_pointer=True,cold_or_stale_capture_fallback4=True,
                    FP32_masters_state_preserved=True''')
ast.parse(guard);(root/'scripts/check_both_c1280_zero_parts.py').write_text(guard)
print('WORLD_FFN1280_ZERO_PREPARED master_key_refresh=true world_only=true FP32_previous=true',flush=True)
