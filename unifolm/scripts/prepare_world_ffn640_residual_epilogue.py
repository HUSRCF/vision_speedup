"""Extend exact residual epilogues to the accepted C640 two-product FFN path."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'


def replace(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


source=(utils/'world_ffn_residual_epilogue.py').read_text()
source=source.replace('ffn_residual_epilogue','ffn640_residual_epilogue')
source=replace(source,'ROWS={320:{40960}}','ROWS={640:{10240}}')
start=source.index('def compatible(block):');end=source.index('\n\ndef make_forward(block):')
source=source[:start]+'''def compatible(block):
    from .world_norm3_packed_geglu import compatible as norm_compatible
    return (norm_compatible(block) and block.norm3.normalized_shape==(640,)
        and getattr(block,'_world_norm3_packed_geglu_installed',False)
        and getattr(block.ff.net[2],'_tf32x3_installed',False)
        and getattr(block.ff.net[2],'_world_two_product_roles_installed',False))
'''+source[end:]
source=source.replace('_ffn_residual_weight_cache','_ffn640_residual_weight_cache')
ast.parse(source);(utils/'world_ffn640_residual_epilogue.py').write_text(source)
path=utils/'inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
source=replace(source,anchor,'''    ffn640_residual_mode=os.environ.get('UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE','0')
    if ffn640_residual_mode not in {'0','1'}:raise ValueError('Invalid C640 FFN residual mode')
    ffn640_residual_count=0
    if ffn640_residual_mode=='1':
        if (norm3_pack_scope!='all' or ffn_residual_mode!='1' or
                os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE')!='output640_ffn640' or
                os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT')!='input'):
            raise ValueError('C640 residual fusion requires verified norm3/C320 and input-protected C640 FFN')
        from .world_ffn640_residual_epilogue import install_world_ffn640_residual_epilogue
        ffn640_residual_count=install_world_ffn640_residual_epilogue(model.model.diffusion_model)
        if ffn640_residual_count!=10:raise ValueError('Unexpected C640 FFN residual count')
    print(f'WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL mode={ffn640_residual_mode} blocks={ffn640_residual_count} '
          'GEMM=previous_two_input_protected_products_BK64_BM64_BN64_group32_stages2 '
          'residual=fp32_before_store static_pack=first_timed_warm_version_refresh_same_storage',flush=True)
'''+anchor)
ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE=${UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE:-0}"\n'
path.write_text(replace(source,anchor,anchor+'    echo "UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE=${UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE:-0}"\n'))
source=(root/'scripts/check_world_ffn_residual_epilogue.py').read_text()
source=source.replace('world_ffn_residual_epilogue','world_ffn640_residual_epilogue')
source=source.replace('install_world_ffn_residual_epilogue','install_world_ffn640_residual_epilogue')
source=source.replace('_world_ffn_residual_epilogue_installed','_world_ffn640_residual_epilogue_installed')
source=source.replace('_ffn_residual_weight_cache','_ffn640_residual_weight_cache')
source=replace(source,'    block.ff.net[2]._tf32x3_installed=True',
    '    block.ff.net[2]._tf32x3_installed=True\n    block.ff.net[2]._world_two_product_roles_installed=True')
source=replace(source,'from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight,half_compensated_linear',
    'from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight\nfrom unifolm_wma.utils.triton_two_product_linear import two_product_linear')
source=replace(source,'''        return half_compensated_linear(value,final.weight,final.bias,
            block_m=64,block_n=64,block_k=64,group_m=32,weight_parts=pack_half_weight(final.weight))''',
    "        return two_product_linear(value,pack_half_weight(final.weight),final.bias,protect='input')")
source=replace(source,'[(320,40960)]','[(640,10240)]')
source=replace(source,"    assert not hasattr(tree.mid,'_world_ffn640_residual_epilogue_installed')",
    "    assert not hasattr(tree.low,'_world_ffn640_residual_epilogue_installed')")
source=source.replace('WORLD_FFN_RESIDUAL_EPILOGUE','WORLD_FFN640_RESIDUAL_EPILOGUE')
ast.parse(source);(root/'scripts/check_world_ffn640_residual_epilogue.py').write_text(source)
# Incremental pair: norm3+C320 already active in both graphs, only C640 changes.
source=(root/'scripts/profile_world_ffn_residual_epilogue_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE','UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE')
source=source.replace('world_ffn_residual_epilogue as fusion_helper','world_ffn640_residual_epilogue as fusion_helper')
source=source.replace('fusion_helper.ffn_residual_epilogue','fusion_helper.ffn640_residual_epilogue')
anchor='''                        from unifolm_wma.utils.world_norm3_packed_geglu import install_world_norm3_packed_geglu
                        assert install_world_norm3_packed_geglu(model.model.diffusion_model)==32
                        assert fusion_helper.install_world_ffn_residual_epilogue(model.model.diffusion_model)==10'''
source=replace(source,anchor,'                        assert fusion_helper.install_world_ffn640_residual_epilogue(model.model.diffusion_model)==10')
source=source.replace('_world_ffn_residual_epilogue_calls','_world_ffn640_residual_epilogue_calls')
source=source.replace('_world_ffn_residual_epilogue_installed','_world_ffn640_residual_epilogue_installed')
source=replace(source,'for width in [320]','for width in [640]')
source=source.replace('WORLD_FFN_RESIDUAL_EPILOGUE_MODEL','WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL')
ast.parse(source);(root/'scripts/profile_world_ffn640_residual_epilogue_pair.py').write_text(source)
source=(root/'scripts/run_world_ffn_residual_epilogue_pair.py').read_text()
source=replace(source,"flags['UNIFOLM_WORLD_NORM3_PACKED_GEGLU']='0'", "flags['UNIFOLM_WORLD_NORM3_PACKED_GEGLU']='all'")
source=replace(source,"flags['UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE']='0'", "flags['UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE']='1'\nflags['UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE']='0'")
source=source.replace('world-ffn-residual-epilogue-graph-pair-','world-ffn640-residual-epilogue-graph-pair-')
source=source.replace('scripts/profile_world_ffn_residual_epilogue_pair.py','scripts/profile_world_ffn640_residual_epilogue_pair.py')
source=source.replace('WORLD_FFN_RESIDUAL_EPILOGUE_MODEL','WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL')
ast.parse(source);(root/'scripts/run_world_ffn640_residual_epilogue_pair.py').write_text(source)
for prefix in ['wait','run']:
    source=(root/f'scripts/{prefix}_world_ffn_residual_epilogue_reserved_pair.sh').read_text()
    source=source.replace('world_ffn_residual_epilogue','world_ffn640_residual_epilogue')
    source=source.replace('world-ffn-residual-epilogue-pair-','world-ffn640-residual-epilogue-pair-')
    source=source.replace('WORLD_FFN_RESIDUAL_EPILOGUE_MODEL','WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL')
    (root/f'scripts/{prefix}_world_ffn640_residual_epilogue_reserved_pair.sh').write_text(source)
print('WORLD_FFN640_RESIDUAL_EPILOGUE_PREPARED ten_C640_blocks=true baseline=norm3_plus_C320 incremental_C640_pair=true',flush=True)
