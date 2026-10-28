"""Install exact C640 decision pipeline into a new immutable snapshot only."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'


def replace(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


source=(utils/'triton_zero_weight_low_geglu.py').read_text();lines=source.splitlines(keepends=True)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='residual_count')
count_source=''.join(lines[node.lineno-1:node.end_lineno])
count_source=replace(count_source,'def residual_count(', 'def _refresh_shared_ge_count(')
count_source=replace(count_source,'    if not selected(parts[0].shape[1]):return None\n','')
new=ast.parse(count_source).body[0]
assert [ast.dump(n,include_attributes=False) for n in node.body[1:]]==[ast.dump(n,include_attributes=False) for n in new.body]
(utils/'decision_shared_ge_count.py').write_text('"""Original GE count/cache refresh without the world-only policy gate."""\nimport torch\n\n'+count_source+'\n')
path=utils/'inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
source=replace(source,anchor,'''    decision_chunked_hidden=os.environ.get('UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS','0')
    if decision_chunked_hidden not in {'0','1'}:raise ValueError('Invalid decision chunked hidden mode')
    decision_chunked_hidden_count=0
    if decision_chunked_hidden=='1':
        if (os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU')!='both'
                or os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU_SCOPE') not in {'all','640'}
                or os.environ.get('UNIFOLM_DECISION_VENDOR_FFN')!='chunked'
                or os.environ.get('UNIFOLM_DECISION_PRIMARY_FFN')!='1280'
                or os.environ.get('UNIFOLM_WORLD_NORM3_PACKED_GEGLU')!='all'
                or os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS')!='1'
                or os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY')!='640'):
            raise ValueError('Decision chunked hidden requires original C640 primary GEGLU, chunk512 FFN and world shared caches')
        from .decision_chunked_hidden_parts import install_decision_chunked_hidden_parts
        decision_chunked_hidden_count=install_decision_chunked_hidden_parts(model.model.diffusion_model)
        if decision_chunked_hidden_count!=10:raise ValueError('Unexpected decision chunked hidden count')
    print(f'DECISION_CHUNKED_HIDDEN_PARTS_MODEL mode={decision_chunked_hidden} blocks={decision_chunked_hidden_count} '
          'norm=previous_FP32_residual_norm GEGLU=previous_both_primary_BK64_BM64_BN64_group32 '
          'hidden=vector_Half_high_low GEMM=previous_chunk512_three_batched_products '
          'FP32_ordered_reduce_bias_residual=true static_weights=shared_world_GE_and_final_caches '
          'weight_pack=first_timed_warm_version_refresh_same_storage dynamic_parts=fresh_each_call '
          'master_public_output=fp32 world=previous',flush=True)
'''+anchor)
ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE=${UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE:-0}"\n'
path.write_text(replace(source,anchor,anchor+'    echo "UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS=${UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS:-0}"\n'))
source=(root/'scripts/profile_world_geglu_hidden_parts_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS','UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS')
source=source.replace('world_geglu_hidden_parts as fusion_helper','decision_chunked_hidden_parts as fusion_helper')
source=source.replace('fusion_helper.packed_hidden_ffn_residual','fusion_helper.chunked_hidden_residual')
source=source.replace('install_world_geglu_hidden_parts(model.model.diffusion_model)==10','install_decision_chunked_hidden_parts(model.model.diffusion_model)==10')
source=source.replace('_world_geglu_hidden_parts_installed','_decision_chunked_hidden_parts_installed')
source=source.replace('_world_geglu_hidden_parts_calls','_decision_chunked_hidden_parts_calls')
source=replace(source,"if options.profile_stage=='world':\n                    assert all(n>=3 for n in tail_dispatch.values())", "if options.profile_stage=='decision':\n                    assert all(n>=3 for n in tail_dispatch.values())")
source=replace(source,"if options.profile_stage=='world':\n                    assert all(fused_calls.get(str(width),0)>=30 for width in [640])", "if options.profile_stage=='decision':\n                    assert all(fused_calls.get(str(width),0)>=30 for width in [640])")
source=replace(source,"                results, entries = {}, {}",'''                results, entries = {}, {}
                shared_before=None''')
anchor="                    if mode!='0':\n                        assert fusion_helper.install_decision_chunked_hidden_parts(model.model.diffusion_model)==10"
source=replace(source,anchor,'''                    if mode!='0':
                        if options.profile_stage=='world':
                            shared_before={name:(tuple(t.data_ptr() for t in m.ff.net[0]._norm3_geglu_weight_cache[1]),m.ff.net[0]._norm3_geglu_weight_cache[2].data_ptr(),tuple(t.data_ptr() for t in m.ff.net[2]._ffn640_residual_weight_cache[1]))
                                for name,m in model.model.diffusion_model.named_modules()
                                if getattr(m,'_world_geglu_hidden_parts_installed',False)}
                            assert len(shared_before)==10
                        assert fusion_helper.install_decision_chunked_hidden_parts(model.model.diffusion_model)==10''')
anchor="                reference, spec = tree_flatten(results['baseline'])"
source=replace(source,anchor,'''                shared_after={name:(tuple(t.data_ptr() for t in m.ff.net[0]._norm3_geglu_weight_cache[1]),m.ff.net[0]._norm3_geglu_weight_cache[2].data_ptr(),tuple(t.data_ptr() for t in m.ff.net[2]._ffn640_residual_weight_cache[1]))
                    for name,m in model.model.diffusion_model.named_modules()
                    if getattr(m,'_decision_chunked_hidden_parts_installed',False)}
                assert len(shared_after)==10
                if shared_before is not None:assert shared_before==shared_after
'''+anchor)
source=replace(source,"                    fused_hidden_pipeline_calls_per_width=dict(fused_calls),", "                    shared_GE_and_final_parts_unchanged_in_world=(shared_before==shared_after if shared_before is not None else None),\n                    shared_cache_count=len(shared_after), fused_hidden_pipeline_calls_per_width=dict(fused_calls),")
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','DECISION_CHUNKED_HIDDEN_PARTS_MODEL')
ast.parse(source);(root/'scripts/profile_decision_chunked_hidden_parts_pair.py').write_text(source)
source=(root/'scripts/run_world_geglu_hidden_parts_pair.py').read_text()
source=replace(source,"flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='0'", "flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='1'\nflags['UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS']='0'")
source=source.replace('world-geglu-hidden-parts-graph-pair-','decision-chunked-hidden-parts-graph-pair-')
source=source.replace('scripts/profile_world_geglu_hidden_parts_pair.py','scripts/profile_decision_chunked_hidden_parts_pair.py')
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','DECISION_CHUNKED_HIDDEN_PARTS_MODEL')
ast.parse(source);(root/'scripts/run_decision_chunked_hidden_parts_pair.py').write_text(source)
for prefix in ['wait','run']:
    source=(root/f'scripts/{prefix}_world_geglu_hidden_parts_reserved_pair.sh').read_text()
    source=source.replace('world_geglu_hidden_parts','decision_chunked_hidden_parts')
    source=source.replace('world-geglu-hidden-parts-pair-','decision-chunked-hidden-parts-pair-')
    source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','DECISION_CHUNKED_HIDDEN_PARTS_MODEL')
    (root/f'scripts/{prefix}_decision_chunked_hidden_parts_reserved_pair.sh').write_text(source)
print('DECISION_CHUNKED_HIDDEN_PARTS_PREPARED original_chunk512_math=true shared_GE_and_final_world_caches=true blocks=10')
