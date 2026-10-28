"""Keep the accepted C320 prefix and replace only the FFN consumer."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'
text=(utils/'world_geglu_hidden_parts_c320.py').read_text()
def change(a,b):
    global text
    assert text.count(a)==1,a
    text=text.replace(a,b)
change('from .geglu_hidden_parts_c320_pipeline import geglu_hidden_parts_c320,packed_hidden_ffn320_residual',
'''from .geglu_hidden_parts_c320_pipeline import geglu_hidden_parts_c320
from .ffn320_zero_residual import zero_ffn320_hidden_residual,shared_final_low_count''')
change("        and getattr(block,'_world_ffn_residual_epilogue_installed',False))",
"        and getattr(block,'_world_ffn_residual_epilogue_installed',False)\n        and getattr(block,'_world_geglu_hidden_parts_c320_installed',False))")
change('        attended=block.attn1(block.norm1(value),',
'''        # Parts are written by Triton out=, so their Tensor._version is not
        # sufficient. Force a count refresh when the original master changes.
        previous=getattr(final,'_world_ffn320_low_count_key',None)
        count=shared_final_low_count(final_cache[1],refresh=(previous!=final_key))
        if count is None:return original(value,context=context,mask=mask)
        final._world_ffn320_low_count_key=final_key
        attended=block.attn1(block.norm1(value),''')
change('result=packed_hidden_ffn320_residual(hidden_parts,final_cache[1],residual.view(-1,c),final.bias)',
       'result=zero_ffn320_hidden_residual(hidden_parts,final_cache[1],count,residual.view(-1,c),final.bias)')
text=text.replace('_world_geglu_hidden_parts_c320_calls','_world_ffn320_zero_calls')
text=text.replace('_world_geglu_hidden_parts_c320_installed', '_world_ffn320_zero_installed')
# compatible must require the old wrapper, whereas the duplicate/factory
# marker is specific to the new wrapper.
text=text.replace("and getattr(block,'_world_ffn320_zero_installed',False))",
                  "and getattr(block,'_world_geglu_hidden_parts_c320_installed',False))")
text=text.replace('install_world_geglu_hidden_parts_c320(', 'install_world_ffn320_zero(')
ast.parse(text);(utils/'world_ffn320_zero.py').write_text(text)

guard=(root/'scripts/check_world_geglu_hidden_parts_c320.py').read_text()
def guard_change(a,b):
    global guard
    assert guard.count(a)==1,a
    guard=guard.replace(a,b)
guard_change('world_geglu_hidden_parts_c320 as helper','world_ffn320_zero as helper')
guard_change('    block._world_ffn_residual_epilogue_installed=True',
             '    block._world_ffn_residual_epilogue_installed=True\n    block._world_geglu_hidden_parts_c320_installed=True')
guard_change('            p.weight.copy_(p.weight.half().float())',
'''            p.weight.copy_(p.weight.half().float())
            final=block.ff.net[2]
            final.weight.copy_(final.weight.half().float())''')
guard_change('                assert int(cache[2])==0',
'''                assert int(cache[2])==0
                final_count=final._ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count[1]
                assert int(final_count)==0
                final_count_pointer=final_count.data_ptr()
                assert final._world_ffn320_low_count_key==final._ffn_residual_weight_cache[0]''')
guard_change('                    block.norm3.weight.mul_(1.001);block.norm3.bias.add_(.0003)',
'''                    for name in ['norm1','norm2','norm3']:
                        norm=getattr(block,name);norm.weight.mul_(1.001);norm.bias.add_(.0003)''')
guard_change('                    block.ff.net[2].weight.add_(.000001)',
'''                    final.weight.add_(.00013)
                    if quantized:final.weight.copy_(final.weight.half().float())''')
guard_change('                    assert (int(cache[2])==0)==quantized',
'''                    assert (int(cache[2])==0)==quantized
                    count=final._ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count[1]
                    assert count.data_ptr()==final_count_pointer and (int(count)==0)==quantized
                    assert final._world_ffn320_low_count_key==final._ffn_residual_weight_cache[0]''')
guard_change("            print('WORLD_GEGLU_HIDDEN_PARTS_C320_RUNTIME '",
"""            # The original world wrapper can refresh the shared parts while
            # the new wrapper is bypassed. Capture must not retain stale count.
            from unifolm_wma.utils import world_geglu_hidden_parts_c320 as previous_helper
            for kind in ['count_missing','GE_stale','final_stale','count_stale_after_old_refresh']:
                extra=fixture(channels,bias);extra._forward=sentinel
                extra_forward=helper.make_forward(extra)
                extra_forward(value,context=context)
                if kind=='count_missing':
                    del extra.ff.net[2]._ffn_residual_weight_cache[1][0]._ffn1280_zero_low_count
                elif kind=='GE_stale':extra.ff.net[0].proj.weight.add_(.00013)
                else:
                    extra.ff.net[2].weight.add_(.00017)
                    if kind=='count_stale_after_old_refresh':
                        previous_helper.make_forward(extra)(value,context=context)
                        assert extra._forward is sentinel
                with patch.object(helper,'pack_half_weight',side_effect=AssertionError('cold capture packed weights')):
                    cg=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(cg):co=extra_forward(value,context=context)
                    cg.replay();exact(co,sentinel(value))
                cg.reset();del extra,extra_forward,cg,co
            print('WORLD_GEGLU_HIDDEN_PARTS_C320_RUNTIME '""")
guard_change('                zero_nonzero_zero_device_predicate=True,fp32_master_storage_state_preserved=True,',
'''                zero_nonzero_zero_device_predicate=True,FFN_zero_nonzero_zero_device_predicate=True,
                FFN_count_master_version_refresh_same_pointer=True,cold_stale_count_capture_fallback4=True,
                fp32_master_storage_state_preserved=True,''')
# Reference marker setup remains old; assertions below inspect only new scope.
guard=guard.replace('helper.install_world_geglu_hidden_parts_c320(', 'helper.install_world_ffn320_zero(')
guard=guard.replace("hasattr(tree.initial,'_world_geglu_hidden_parts_c320_installed')", "hasattr(tree.initial,'_world_ffn320_zero_installed')")
guard=guard.replace("hasattr(tree.mid,'_world_geglu_hidden_parts_c320_installed')", "hasattr(tree.mid,'_world_ffn320_zero_installed')")
guard=guard.replace("hasattr(tree.wide,'_world_geglu_hidden_parts_c320_installed')", "hasattr(tree.wide,'_world_ffn320_zero_installed')")
guard=guard.replace("hasattr(tree.outside,'_world_geglu_hidden_parts_c320_installed')", "hasattr(tree.outside,'_world_ffn320_zero_installed')")
guard=guard.replace('WORLD_GEGLU_HIDDEN_PARTS_C320_RUNTIME','WORLD_FFN320_ZERO_RUNTIME')
ast.parse(guard);(root/'scripts/check_world_ffn320_zero.py').write_text(guard)
print('WORLD_FFN320_ZERO_PREPARED same_shader=true original_master_key_refresh=true first_timed_warm=true',flush=True)
