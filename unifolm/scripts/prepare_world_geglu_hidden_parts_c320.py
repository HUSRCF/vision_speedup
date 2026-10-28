"""Integrate C320 fresh hidden parts while preserving three-product FFN math."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'


def replace(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


source=(utils/'world_ffn_residual_epilogue.py').read_text()
source=replace(source,'from .geglu_from_input_parts import geglu_from_input_parts\nfrom .ffn_residual_epilogue import ffn_residual_epilogue',
    'from .geglu_hidden_parts_c320_pipeline import geglu_hidden_parts_c320,packed_hidden_ffn320_residual')
start=source.index('def compatible(block):');end=source.index('\n\ndef make_forward(block):')
source=source[:start]+'''def compatible(block):
    from .world_ffn_residual_epilogue import compatible as previous_compatible
    return (previous_compatible(block)
        and getattr(block,'_world_ffn_residual_epilogue_installed',False))
'''+source[end:]
source=replace(source,'''        hidden=geglu_from_input_parts((high,low),cache[1],cache[2],p.bias,
                                     shape=(*value.shape[:-1],c*4))''',
    '        hidden_parts=geglu_hidden_parts_c320((high,low),cache[1],cache[2],p.bias)')
source=replace(source,'        return ffn_residual_epilogue(block.ff.net[1](hidden),final_cache[1],residual,final.bias)',
    '''        # Eval Dropout is the identity; its class and every training flag
        # are validated before skipping the intermediate FP32 tensor.
        result=packed_hidden_ffn320_residual(hidden_parts,final_cache[1],residual.view(-1,c),final.bias)
        block._world_geglu_hidden_parts_c320_calls=getattr(block,'_world_geglu_hidden_parts_c320_calls',0)+1
        return result.view_as(value)''')
start=source.index('def install_world_ffn_residual_epilogue')
source=source[:start]+source[start:].replace('install_world_ffn_residual_epilogue','install_world_geglu_hidden_parts_c320').replace('_world_ffn_residual_epilogue_installed','_world_geglu_hidden_parts_c320_installed')
ast.parse(source);(utils/'world_geglu_hidden_parts_c320.py').write_text(source)
path=utils/'inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
source=replace(source,anchor,'''    world_hidden_parts_c320=os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS_C320','0')
    if world_hidden_parts_c320 not in {'0','1'}:raise ValueError('Invalid C320 hidden parts mode')
    world_hidden_parts_c320_count=0
    if world_hidden_parts_c320=='1':
        if (world_hidden_parts!='1' or norm3_pack_scope!='all' or ffn_residual_mode!='1' or
                os.environ.get('UNIFOLM_WORLD_C320_BK64_GEGLU')!='gate'):
            raise ValueError('C320 hidden parts requires original three-product FFN and C640 pipeline')
        from .world_geglu_hidden_parts_c320 import install_world_geglu_hidden_parts_c320
        world_hidden_parts_c320_count=install_world_geglu_hidden_parts_c320(model.model.diffusion_model)
        if world_hidden_parts_c320_count!=10:raise ValueError('Unexpected C320 hidden parts count')
    print(f'WORLD_GEGLU_HIDDEN_PARTS_C320_MODEL mode={world_hidden_parts_c320} blocks={world_hidden_parts_c320_count} '
          'producer=previous_GEGLU_group16_vector_Half_store consumer=previous_three_products_BK64_BM64_BN64_group32 '
          'dynamic_hidden_parts=fresh_each_call static_weights=previous_first_timed_warm_version_refresh '
          'master_public_output=fp32 decision=previous',flush=True)
'''+anchor)
ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS=${UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS:-0}"\n'
path.write_text(replace(source,anchor,anchor+'    echo "UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS_C320=${UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS_C320:-0}"\n'))
source=(root/'scripts/check_world_ffn_residual_epilogue.py').read_text()
source=source.replace('world_ffn_residual_epilogue as helper','world_geglu_hidden_parts_c320 as helper')
source=source.replace('install_world_ffn_residual_epilogue','install_world_geglu_hidden_parts_c320')
source=source.replace('_world_ffn_residual_epilogue_installed','_world_geglu_hidden_parts_c320_installed')
source=replace(source,'    block._world_norm3_packed_geglu_installed=True',
    '    block._world_norm3_packed_geglu_installed=True\n    block._world_ffn_residual_epilogue_installed=True')
source=source.replace('WORLD_FFN_RESIDUAL_EPILOGUE_RUNTIME','WORLD_GEGLU_HIDDEN_PARTS_C320_RUNTIME')
ast.parse(source);(root/'scripts/check_world_geglu_hidden_parts_c320.py').write_text(source)
source=(root/'scripts/profile_world_geglu_hidden_parts_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS','UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS_C320')
source=source.replace('world_geglu_hidden_parts as fusion_helper','world_geglu_hidden_parts_c320 as fusion_helper')
source=source.replace('fusion_helper.packed_hidden_ffn_residual','fusion_helper.packed_hidden_ffn320_residual')
source=source.replace('install_world_geglu_hidden_parts(', 'install_world_geglu_hidden_parts_c320(')
source=source.replace('_world_geglu_hidden_parts_installed','_world_geglu_hidden_parts_c320_installed')
source=source.replace('_world_geglu_hidden_parts_calls','_world_geglu_hidden_parts_c320_calls')
source=replace(source,'for width in [640]','for width in [320]')
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','WORLD_GEGLU_HIDDEN_PARTS_C320_MODEL')
ast.parse(source);(root/'scripts/profile_world_geglu_hidden_parts_c320_pair.py').write_text(source)
source=(root/'scripts/run_world_geglu_hidden_parts_pair.py').read_text()
source=replace(source,"flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='0'",
    "flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='1'\nflags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS_C320']='0'")
source=source.replace('world-geglu-hidden-parts-graph-pair-','world-geglu-hidden-parts-c320-graph-pair-')
source=source.replace('scripts/profile_world_geglu_hidden_parts_pair.py','scripts/profile_world_geglu_hidden_parts_c320_pair.py')
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','WORLD_GEGLU_HIDDEN_PARTS_C320_MODEL')
ast.parse(source);(root/'scripts/run_world_geglu_hidden_parts_c320_pair.py').write_text(source)
for prefix in ['wait','run']:
    source=(root/f'scripts/{prefix}_world_geglu_hidden_parts_reserved_pair.sh').read_text()
    source=source.replace('world_geglu_hidden_parts','world_geglu_hidden_parts_c320')
    source=source.replace('world-geglu-hidden-parts-pair-','world-geglu-hidden-parts-c320-pair-')
    source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','WORLD_GEGLU_HIDDEN_PARTS_C320_MODEL')
    (root/f'scripts/{prefix}_world_geglu_hidden_parts_c320_reserved_pair.sh').write_text(source)
print('WORLD_GEGLU_HIDDEN_PARTS_C320_PREPARED original_three_product_FFN=true blocks=10 baseline_C640_pipeline=true',flush=True)
