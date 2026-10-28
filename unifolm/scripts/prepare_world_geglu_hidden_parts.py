"""Install the vector Half hidden pipeline on the verified C640 world blocks."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'


def replace(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


source=(utils/'world_ffn640_residual_epilogue.py').read_text()
source=replace(source,'from .geglu_from_input_parts import geglu_from_input_parts\nfrom .ffn640_residual_epilogue import ffn640_residual_epilogue',
    'from .geglu_hidden_parts_pipeline import geglu_hidden_parts,packed_hidden_ffn_residual')
start=source.index('def compatible(block):');end=source.index('\n\ndef make_forward(block):')
source=source[:start]+'''def compatible(block):
    from .world_ffn640_residual_epilogue import compatible as previous_compatible
    return (previous_compatible(block)
        and getattr(block,'_world_ffn640_residual_epilogue_installed',False))
'''+source[end:]
source=replace(source,'''        hidden=geglu_from_input_parts((high,low),cache[1],cache[2],p.bias,
                                     shape=(*value.shape[:-1],c*4))''',
    '        hidden_parts=geglu_hidden_parts((high,low),cache[1],cache[2],p.bias)')
source=replace(source,'        return ffn640_residual_epilogue(block.ff.net[1](hidden),final_cache[1],residual,final.bias)',
    '''        # Dropout is the identity here: compatible() requires Dropout and
        # the dispatch guard rejects every training module.
        result=packed_hidden_ffn_residual(hidden_parts,final_cache[1],residual.view(-1,c),final.bias)
        block._world_geglu_hidden_parts_calls=getattr(block,'_world_geglu_hidden_parts_calls',0)+1
        return result.view_as(value)''')
start=source.index('def install_world_ffn640_residual_epilogue')
source=source[:start]+source[start:].replace('install_world_ffn640_residual_epilogue','install_world_geglu_hidden_parts').replace('_world_ffn640_residual_epilogue_installed','_world_geglu_hidden_parts_installed')
ast.parse(source);(utils/'world_geglu_hidden_parts.py').write_text(source)
path=utils/'inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
source=replace(source,anchor,'''    world_hidden_parts=os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS','0')
    if world_hidden_parts not in {'0','1'}:raise ValueError('Invalid GEGLU hidden parts mode')
    world_hidden_parts_count=0
    if world_hidden_parts=='1':
        if (norm3_pack_scope!='all' or ffn640_residual_mode!='1' or
                os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY')!='640' or
                os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE')!='output640_ffn640' or
                os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT')!='input'):
            raise ValueError('GEGLU hidden parts requires verified C640 primary gate and protected FFN')
        from .world_geglu_hidden_parts import install_world_geglu_hidden_parts
        world_hidden_parts_count=install_world_geglu_hidden_parts(model.model.diffusion_model)
        if world_hidden_parts_count!=10:raise ValueError('Unexpected C640 hidden parts count')
    print(f'WORLD_GEGLU_HIDDEN_PARTS_MODEL mode={world_hidden_parts} blocks={world_hidden_parts_count} '
          'producer=previous_GEGLU_FP32_math_vector_Half_store consumer=previous_two_protected_products_BN128 '
          'dynamic_hidden_parts=fresh_each_call static_weights=previous_first_timed_warm_version_refresh '
          'master_public_output=fp32 decision=previous',flush=True)
'''+anchor)
ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE=${UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE:-0}"\n'
path.write_text(replace(source,anchor,anchor+'    echo "UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS=${UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS:-0}"\n'))
source=(root/'scripts/check_world_ffn640_residual_epilogue.py').read_text()
source=source.replace('world_ffn640_residual_epilogue as helper','world_geglu_hidden_parts as helper')
source=source.replace('install_world_ffn640_residual_epilogue','install_world_geglu_hidden_parts')
source=source.replace('_world_ffn640_residual_epilogue_installed','_world_geglu_hidden_parts_installed')
source=replace(source,'    block._world_norm3_packed_geglu_installed=True',
    '    block._world_norm3_packed_geglu_installed=True\n    block._world_ffn640_residual_epilogue_installed=True')
source=source.replace('WORLD_FFN640_RESIDUAL_EPILOGUE_RUNTIME','WORLD_GEGLU_HIDDEN_PARTS_RUNTIME')
ast.parse(source);(root/'scripts/check_world_geglu_hidden_parts.py').write_text(source)
source=(root/'scripts/profile_world_ffn640_residual_epilogue_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE','UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS')
source=source.replace('world_ffn640_residual_epilogue as fusion_helper','world_geglu_hidden_parts as fusion_helper')
source=source.replace('fusion_helper.ffn640_residual_epilogue','fusion_helper.packed_hidden_ffn_residual')
source=source.replace('install_world_ffn640_residual_epilogue','install_world_geglu_hidden_parts')
source=source.replace('_world_ffn640_residual_epilogue_installed','_world_geglu_hidden_parts_installed')
source=source.replace('_world_ffn640_residual_epilogue_calls','_world_geglu_hidden_parts_calls')
source=source.replace('WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL','WORLD_GEGLU_HIDDEN_PARTS_MODEL')
source=source.replace('fused_tail_calls_per_width','fused_hidden_pipeline_calls_per_width')
source=source.replace('tail_dispatch_per_module','hidden_parts_dispatch_per_module')
ast.parse(source);(root/'scripts/profile_world_geglu_hidden_parts_pair.py').write_text(source)
source=(root/'scripts/run_world_ffn640_residual_epilogue_pair.py').read_text()
source=replace(source,"flags['UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE']='0'",
    "flags['UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE']='1'\nflags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='0'\nassert flags['UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE']=='1'")
source=source.replace('world-ffn640-residual-epilogue-graph-pair-','world-geglu-hidden-parts-graph-pair-')
source=source.replace('scripts/profile_world_ffn640_residual_epilogue_pair.py','scripts/profile_world_geglu_hidden_parts_pair.py')
source=source.replace('WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL','WORLD_GEGLU_HIDDEN_PARTS_MODEL')
ast.parse(source);(root/'scripts/run_world_geglu_hidden_parts_pair.py').write_text(source)
for prefix in ['wait','run']:
    source=(root/f'scripts/{prefix}_world_ffn640_residual_epilogue_reserved_pair.sh').read_text()
    source=source.replace('world_ffn640_residual_epilogue','world_geglu_hidden_parts')
    source=source.replace('world-ffn640-residual-epilogue-pair-','world-geglu-hidden-parts-pair-')
    source=source.replace('WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL','WORLD_GEGLU_HIDDEN_PARTS_MODEL')
    (root/f'scripts/{prefix}_world_geglu_hidden_parts_reserved_pair.sh').write_text(source)
print('WORLD_GEGLU_HIDDEN_PARTS_PREPARED blocks=10 vector_store=true dynamic_parts=fresh weights=previous_cache FP32_public_output=true',flush=True)
