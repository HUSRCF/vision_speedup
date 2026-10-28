"""Generate a C320 FFN-tail wrapper from the verified norm3 wrapper."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1])
utils=root/'src/unifolm_wma/utils'


def replace(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


source=(utils/'world_norm3_packed_geglu.py').read_text()
source=replace(source,'ROWS={320:{40960},640:{10240},1280:{2560,640}}','ROWS={320:{40960}}')
source=replace(source,'from .geglu_from_input_parts import geglu_from_input_parts',
    'from .geglu_from_input_parts import geglu_from_input_parts\nfrom .ffn_residual_epilogue import ffn_residual_epilogue')
source=replace(source,"    if c==320:return getattr(ge,'_world_bk64_c320_geglu_installed',False)",
    "    if c==320:return (getattr(ge,'_world_bk64_c320_geglu_installed',False)\n"
    "        and getattr(block,'_world_norm3_packed_geglu_installed',False)\n"
    "        and getattr(block.ff.net[2],'_tf32x3_installed',False)\n"
    "        and not getattr(block.ff.net[2],'_world_two_product_roles_installed',False))")
source=replace(source,'        parameters=[p.weight,',
    '        final=block.ff.net[2]\n        parameters=[final.weight,*([final.bias] if final.bias is not None else []),p.weight,')
# Check both cache keys before doing any work; a cold capture must fall back.
anchor="        cache=getattr(ge,'_norm3_geglu_weight_cache',None)\n"
source=replace(source,anchor,'''        fw=final.weight
        try:final_key=(fw.data_ptr(),fw._version,tuple(fw.shape),fw.dtype,fw.device)
        except RuntimeError:return original(value,context=context,mask=mask)
        final_cache=getattr(final,'_ffn_residual_weight_cache',None)
        if (final_cache is None or final_cache[0]!=final_key) and torch.cuda.is_current_stream_capturing():
            return original(value,context=context,mask=mask)
'''+anchor)
anchor='        attended=block.attn1(block.norm1(value),\n'
source=replace(source,anchor,'''        if final_cache is None or final_cache[0]!=final_key:
            reuse=(final_cache[1] if final_cache is not None and final_cache[1][0].shape==fw.shape
                   and final_cache[1][0].device==fw.device else None)
            final_cache=final_key,pack_half_weight(fw,out=reuse)
            final._ffn_residual_weight_cache=final_cache
'''+anchor)
source=replace(source,'        return block.ff.net[2](block.ff.net[1](hidden))+residual',
    "        block._world_ffn_residual_epilogue_calls=getattr(block,'_world_ffn_residual_epilogue_calls',0)+1\n"
    '        return ffn_residual_epilogue(block.ff.net[1](hidden),final_cache[1],residual,final.bias)')
source=source.replace('install_world_norm3_packed_geglu','install_world_ffn_residual_epilogue')
source=replace(source,"getattr(block,'_world_norm3_packed_geglu_installed',False):\n            raise",
    "getattr(block,'_world_ffn_residual_epilogue_installed',False):\n            raise")
source=replace(source,'        block._world_norm3_packed_geglu_installed=True',
    '        block._world_ffn_residual_epilogue_installed=True')
ast.parse(source);(utils/'world_ffn_residual_epilogue.py').write_text(source)

path=utils/'inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
addition='''    ffn_residual_mode=os.environ.get('UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE','0')
    if ffn_residual_mode not in {'0','1'}:raise ValueError('Invalid FFN residual epilogue mode')
    ffn_residual_count=0
    if ffn_residual_mode=='1':
        required={'UNIFOLM_WORLD_NORM3_PACKED_GEGLU':'all',
            'UNIFOLM_COMPENSATED_LINEAR_BACKEND':'scaled-fp16',
            'UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE':'1',
            'UNIFOLM_COMPENSATED_LINEAR_GROUP_M':'32',
            'UNIFOLM_COMPENSATED_LINEAR_BLOCK_K':'64',
            'UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK':'wide',
            'UNIFOLM_TF32X3_NARROW_OUTPUT':'1'}
        if any(os.environ.get(k)!=v for k,v in required.items()):
            raise ValueError('FFN residual epilogue requires verified world norm3 and original C320 linear policy')
        from .world_ffn_residual_epilogue import install_world_ffn_residual_epilogue
        ffn_residual_count=install_world_ffn_residual_epilogue(model.model.diffusion_model)
        if ffn_residual_count!=10:raise ValueError('Unexpected C320 FFN residual epilogue count')
    print(f'WORLD_FFN_RESIDUAL_EPILOGUE_MODEL mode={ffn_residual_mode} blocks={ffn_residual_count} '
          'GEMM=previous_three_products_BK64_BM64_BN64_group32_stages2 '
          'residual_add=fp32_before_store weight_pack=first_timed_warm_version_refresh_same_storage '
          'decision=previous masters=fp32',flush=True)
'''
source=replace(source,anchor,addition+anchor);ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_WORLD_NORM3_PACKED_GEGLU=${UNIFOLM_WORLD_NORM3_PACKED_GEGLU:-0}"\n'
path.write_text(replace(source,anchor,anchor+'    echo "UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE=${UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE:-0}"\n'))

# Real block guard retains all prior dispatch, storage and graph checks.
source=(root/'scripts/check_world_norm3_packed_geglu.py').read_text()
source=source.replace('world_norm3_packed_geglu as helper','world_ffn_residual_epilogue as helper')
source=replace(source,'import pack_half_weight\n','import pack_half_weight,half_compensated_linear\n')
source=replace(source,'    block._inference_fused_residual_norm=True',
    '    block._inference_fused_residual_norm=True\n'
    '    block._world_norm3_packed_geglu_installed=True\n'
    '    block.ff.net[2]._tf32x3_installed=True')
source=replace(source,'    ge.forward=original_geglu\n', '''    ge.forward=original_geglu
    final=block.ff.net[2]
    @torch.no_grad()
    def original_final(value):
        return half_compensated_linear(value,final.weight,final.bias,
            block_m=64,block_n=64,block_k=64,group_m=32,weight_parts=pack_half_weight(final.weight))
    final.forward=original_final
''')
source=replace(source,'[(320,40960),(640,10240),(1280,2560),(1280,640)]','[(320,40960)]')
source=source.replace('assert len(packs)==1','assert len(packs)==2').replace('assert len(packs)==3','assert len(packs)==6')
anchor='                assert int(cache[2])==0\n'
source=replace(source,anchor,anchor+"                final_pointers=tuple(t.data_ptr() for t in block.ff.net[2]._ffn_residual_weight_cache[1])\n")
anchor='                    assert (int(cache[2])==0)==quantized\n'
source=replace(source,anchor,anchor+"                    assert final_pointers==tuple(t.data_ptr() for t in block.ff.net[2]._ffn_residual_weight_cache[1])\n")
anchor="            assert not hasattr(cold.ff.net[0],'_norm3_geglu_weight_cache')\n"
source=replace(source,anchor,anchor+'''            assert not hasattr(cold.ff.net[2],'_ffn_residual_weight_cache')
            from unifolm_wma.utils import world_norm3_packed_geglu as norm_helper
            norm_helper.make_forward(cold)(value)
            assert hasattr(cold.ff.net[0],'_norm3_geglu_weight_cache')
            assert not hasattr(cold.ff.net[2],'_ffn_residual_weight_cache')
            with patch.object(helper,'pack_half_weight',side_effect=AssertionError('cold final cache packed in capture')):
                finalcoldgraph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(finalcoldgraph):finalcoldout=cold_forward(value)
                finalcoldgraph.replay();exact(finalcoldout,sentinel(value))
            assert not hasattr(cold.ff.net[2],'_ffn_residual_weight_cache')
''')
source=source.replace('WORLD_NORM3_PACKED_GEGLU_RUNTIME','WORLD_FFN_RESIDUAL_EPILOGUE_RUNTIME')
source=source.replace('install_world_norm3_packed_geglu(tree)==3','install_world_ffn_residual_epilogue(tree)==1')
source=source.replace('helper.install_world_norm3_packed_geglu(tree)','helper.install_world_ffn_residual_epilogue(tree)')
source=replace(source,"    assert not hasattr(tree.initial,'_world_norm3_packed_geglu_installed')", "    assert not hasattr(tree.initial,'_world_ffn_residual_epilogue_installed')\n    assert not hasattr(tree.mid,'_world_ffn_residual_epilogue_installed')\n    assert not hasattr(tree.wide,'_world_ffn_residual_epilogue_installed')")
source=source.replace("hasattr(tree.outside,'_world_norm3_packed_geglu_installed')","hasattr(tree.outside,'_world_ffn_residual_epilogue_installed')")
source=source.replace('fixtures=8','fixtures=2').replace('cache_first_timed_warm_once=True','both_caches_first_timed_warm_once=True').replace('cache_version_refresh_same_storage=True','both_caches_version_refresh_same_storage=True')
source=replace(source,'native_fallbacks=11,cold_capture_fallback=True,','native_fallbacks=11,cold_capture_fallback=True,final_only_cold_capture_fallback=True,')
ast.parse(source);(root/'scripts/check_world_ffn_residual_epilogue.py').write_text(source)

# Complete graphs compare accepted canonical (norm3 off) against both fusions.
source=(root/'scripts/profile_world_norm3_packed_geglu_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_NORM3_PACKED_GEGLU','UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE')
source=source.replace('world_norm3_packed_geglu as fusion_helper','world_ffn_residual_epilogue as fusion_helper')
source=source.replace('fusion_helper.residual_norm_half_pack','fusion_helper.ffn_residual_epilogue')
source=source.replace("('candidate','all')","('candidate','1')")
source=replace(source,'width=str(value.shape[-1]);','width=str(args[1].shape[-1]);')
source=replace(source,'                        assert fusion_helper.install_world_norm3_packed_geglu(model.model.diffusion_model)==32',
    '                        from unifolm_wma.utils.world_norm3_packed_geglu import install_world_norm3_packed_geglu\n'
    '                        assert install_world_norm3_packed_geglu(model.model.diffusion_model)==32\n'
    '                        assert fusion_helper.install_world_ffn_residual_epilogue(model.model.diffusion_model)==10')
anchor="                if options.profile_stage=='world':\n                    assert all(n>=3 for n in module_dispatch.values()),module_dispatch\n"
source=replace(source,anchor,'''                tail_dispatch={name:getattr(m,'_world_ffn_residual_epilogue_calls',0)
                    for name,m in model.model.diffusion_model.named_modules()
                    if getattr(m,'_world_ffn_residual_epilogue_installed',False)}
                assert len(tail_dispatch)==10
                if options.profile_stage=='world':
                    assert all(n>=3 for n in tail_dispatch.values()),tail_dispatch
                else:
                    assert not any(tail_dispatch.values()),tail_dispatch
'''+anchor)
source=replace(source,'for width in [320,640,1280]','for width in [320]')
source=source.replace('fused_norm3_calls_per_width=dict(fused_calls),','fused_tail_calls_per_width=dict(fused_calls), tail_dispatch_per_module=tail_dispatch,')
source=source.replace('WORLD_NORM3_PACKED_GEGLU_MODEL','WORLD_FFN_RESIDUAL_EPILOGUE_MODEL')
ast.parse(source);(root/'scripts/profile_world_ffn_residual_epilogue_pair.py').write_text(source)
source=(root/'scripts/run_world_norm3_packed_geglu_pair.py').read_text()
source=source.replace('world-norm3-packed-geglu-graph-pair-','world-ffn-residual-epilogue-graph-pair-')
source=source.replace('scripts/profile_world_norm3_packed_geglu_pair.py','scripts/profile_world_ffn_residual_epilogue_pair.py')
source=source.replace('WORLD_NORM3_PACKED_GEGLU_MODEL','WORLD_FFN_RESIDUAL_EPILOGUE_MODEL')
anchor="flags['UNIFOLM_WORLD_NORM3_PACKED_GEGLU']='0'\n"
source=replace(source,anchor,anchor+"flags['UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE']='0'\n")
ast.parse(source);(root/'scripts/run_world_ffn_residual_epilogue_pair.py').write_text(source)
for prefix in ['wait','run']:
    source=(root/f'scripts/{prefix}_world_norm3_packed_geglu_reserved_pair.sh').read_text()
    source=source.replace('world_norm3_packed_geglu','world_ffn_residual_epilogue')
    source=source.replace('world-norm3-packed-geglu-pair-','world-ffn-residual-epilogue-pair-')
    source=source.replace('WORLD_NORM3_PACKED_GEGLU_MODEL','WORLD_FFN_RESIDUAL_EPILOGUE_MODEL')
    (root/f'scripts/{prefix}_world_ffn_residual_epilogue_reserved_pair.sh').write_text(source)
print('WORLD_FFN_RESIDUAL_EPILOGUE_PREPARED scope=10_C320_blocks original_three_product_GEMM=true norm3=32_original_fusion',flush=True)
