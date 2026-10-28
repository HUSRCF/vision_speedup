"""Install the proved decision pipeline without changing any accepted JIT."""
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
source=replace(source,anchor,'''    decision_hidden_parts=os.environ.get('UNIFOLM_DECISION_PRIMARY_HIDDEN_PARTS','0')
    if decision_hidden_parts not in {'0','1'}:raise ValueError('Invalid decision primary hidden mode')
    decision_hidden_parts_count=0
    if decision_hidden_parts=='1':
        if (os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU')!='both'
                or os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU_SCOPE') not in {'all','1280'}
                or os.environ.get('UNIFOLM_DECISION_PRIMARY_FFN') not in {'all','1280'}
                or os.environ.get('UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE')!='1'
                or os.environ.get('UNIFOLM_WORLD_NORM3_PACKED_GEGLU')!='all'
                or os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY')!='640'):
            raise ValueError('Decision primary hidden requires original primary arithmetic and shared world GEGLU cache')
        from .decision_primary_hidden_parts import install_decision_primary_hidden_parts
        decision_hidden_parts_count=install_decision_primary_hidden_parts(model.model.diffusion_model)
        if decision_hidden_parts_count!=12:raise ValueError('Unexpected decision primary hidden count')
    print(f'DECISION_PRIMARY_HIDDEN_PARTS_MODEL mode={decision_hidden_parts} blocks={decision_hidden_parts_count} '
          'norm=previous_FP32_residual_norm GEGLU=previous_both_primary_BK64_BM64_BN32_group32 '
          'hidden=vector_Half_high GEMM=previous_cublas_primary FP32_finish_bias_residual=previous '
          'GEGLU_weight_parts=shared_world_cache final_weight_parts=shared_decision_cache '
          'weight_pack=first_timed_warm_version_refresh_same_storage dynamic_parts=fresh_each_call '
          'master_public_output=fp32 world=previous',flush=True)
'''+anchor)
ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE=${UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE:-0}"\n'
path.write_text(replace(source,anchor,anchor+'    echo "UNIFOLM_DECISION_PRIMARY_HIDDEN_PARTS=${UNIFOLM_DECISION_PRIMARY_HIDDEN_PARTS:-0}"\n'))
source=(root/'scripts/profile_world_geglu_hidden_parts_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS','UNIFOLM_DECISION_PRIMARY_HIDDEN_PARTS')
source=source.replace('world_geglu_hidden_parts as fusion_helper','decision_primary_hidden_parts as fusion_helper')
source=source.replace('fusion_helper.packed_hidden_ffn_residual','fusion_helper.primary_half_input_residual')
source=source.replace('install_world_geglu_hidden_parts(model.model.diffusion_model)==10','install_decision_primary_hidden_parts(model.model.diffusion_model)==12')
source=source.replace('_world_geglu_hidden_parts_installed','_decision_primary_hidden_parts_installed')
source=source.replace('_world_geglu_hidden_parts_calls','_decision_primary_hidden_parts_calls')
source=replace(source,'assert len(tail_dispatch)==10','assert len(tail_dispatch)==12')
source=replace(source,"if options.profile_stage=='world':\n                    assert all(n>=3 for n in tail_dispatch.values())", "if options.profile_stage=='decision':\n                    assert all(n>=3 for n in tail_dispatch.values())")
source=replace(source,"if options.profile_stage=='world':\n                    assert all(fused_calls.get(str(width),0)>=30 for width in [640])", "if options.profile_stage=='decision':\n                    assert all(fused_calls.get(str(width),0)>=36 for width in [1280])")
source=replace(source,"                results, entries = {}, {}",'''                results, entries = {}, {}
                shared_final_before=None; shared_ge_before=None''')
anchor="                    if mode!='0':\n                        assert fusion_helper.install_decision_primary_hidden_parts(model.model.diffusion_model)==12"
source=replace(source,anchor,'''                    if mode!='0':
                        shared_final_before={name:tuple(t.data_ptr() for t in m.ff.net[2]._decision_ffn_residual_weight_cache[1])
                            for name,m in model.model.diffusion_model.named_modules()
                            if getattr(m,'_decision_ffn_residual_epilogue_installed',False)}
                        assert len(shared_final_before)==12
                        if options.profile_stage=='world':
                            shared_ge_before={name:(tuple(t.data_ptr() for t in m.ff.net[0]._norm3_geglu_weight_cache[1]),m.ff.net[0]._norm3_geglu_weight_cache[2].data_ptr())
                                for name,m in model.model.diffusion_model.named_modules()
                                if getattr(m,'_decision_ffn_residual_epilogue_installed',False)}
                            assert len(shared_ge_before)==12
                        assert fusion_helper.install_decision_primary_hidden_parts(model.model.diffusion_model)==12''')
anchor="                reference, spec = tree_flatten(results['baseline'])"
source=replace(source,anchor,'''                shared_final_after={name:tuple(t.data_ptr() for t in m.ff.net[2]._decision_ffn_residual_weight_cache[1])
                    for name,m in model.model.diffusion_model.named_modules()
                    if getattr(m,'_decision_primary_hidden_parts_installed',False)}
                assert shared_final_before==shared_final_after
                shared_ge_after={name:(tuple(t.data_ptr() for t in m.ff.net[0]._norm3_geglu_weight_cache[1]),m.ff.net[0]._norm3_geglu_weight_cache[2].data_ptr())
                    for name,m in model.model.diffusion_model.named_modules()
                    if getattr(m,'_decision_primary_hidden_parts_installed',False)}
                assert len(shared_ge_after)==12
                if shared_ge_before is not None:assert shared_ge_before==shared_ge_after
'''+anchor)
source=replace(source,"                    fused_hidden_pipeline_calls_per_width=dict(fused_calls),", "                    shared_decision_final_parts_unchanged=True, shared_GEGLU_parts_unchanged_in_world=(shared_ge_before==shared_ge_after if shared_ge_before is not None else None),\n                    shared_GEGLU_cache_count=len(shared_ge_after), fused_hidden_pipeline_calls_per_width=dict(fused_calls),")
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','DECISION_PRIMARY_HIDDEN_PARTS_MODEL')
ast.parse(source);(root/'scripts/profile_decision_primary_hidden_parts_pair.py').write_text(source)
source=(root/'scripts/run_world_geglu_hidden_parts_pair.py').read_text()
source=replace(source,"flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='0'", "flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='1'\nflags['UNIFOLM_DECISION_PRIMARY_HIDDEN_PARTS']='0'")
source=source.replace('world-geglu-hidden-parts-graph-pair-','decision-primary-hidden-parts-graph-pair-')
source=source.replace('scripts/profile_world_geglu_hidden_parts_pair.py','scripts/profile_decision_primary_hidden_parts_pair.py')
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','DECISION_PRIMARY_HIDDEN_PARTS_MODEL')
ast.parse(source);(root/'scripts/run_decision_primary_hidden_parts_pair.py').write_text(source)
for prefix in ['wait','run']:
 source=(root/f'scripts/{prefix}_world_geglu_hidden_parts_reserved_pair.sh').read_text()
 source=source.replace('world_geglu_hidden_parts','decision_primary_hidden_parts')
 source=source.replace('world-geglu-hidden-parts-pair-','decision-primary-hidden-parts-pair-')
 source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','DECISION_PRIMARY_HIDDEN_PARTS_MODEL')
 (root/f'scripts/{prefix}_decision_primary_hidden_parts_reserved_pair.sh').write_text(source)
print('DECISION_PRIMARY_HIDDEN_PARTS_PREPARED original_primary_GEGLU_cublas_math=true shared_GE_world_cache=true blocks=12')
