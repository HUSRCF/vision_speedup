"""Install the independently proved native initial hidden pipeline in a clone."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'


def replace(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


path=utils/'initial_native_hidden_parts_pipeline.py';source=path.read_text()
source=replace(source,'_bk32_native_hidden_parts[(triton.cdiv(rows,64)*triton.cdiv(channels,32),)](',
    '_bk32_native_hidden_parts[(triton.cdiv(rows,64)*triton.cdiv(channels,64),)](')
source=replace(source,'rows,channels,inner,bias is not None,64,32,32,False,',
    'rows,channels,inner,bias is not None,64,64,32,False,')
source=replace(source,'_prepacked_gemm[(triton.cdiv(rows,64)*triton.cdiv(channels,64),)](',
    '_prepacked_gemm[(triton.cdiv(rows,64)*triton.cdiv(channels,128),)](')
source=replace(source,'''        rows,channels,inner,bias is not None,64,64,32,32,1,
        num_warps=4,num_stages=2,enable_fp_fusion=False)''',
    '''        rows,channels,inner,bias is not None,64,128,32,32,1,
        num_warps=4,num_stages=3,enable_fp_fusion=False)''')
ast.parse(source);path.write_text(source)
path=utils/'inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
source=replace(source,anchor,'''    initial_hidden_parts=os.environ.get('UNIFOLM_WORLD_INITIAL_NATIVE_HIDDEN_PARTS','0')
    if initial_hidden_parts not in {'0','1'}:raise ValueError('Invalid world initial native hidden mode')
    initial_hidden_parts_count=0
    if initial_hidden_parts=='1':
        if (os.environ.get('UNIFOLM_INITIAL_BK32_NATIVE_GEGLU')!='1'
                or os.environ.get('UNIFOLM_HALF_HIDDEN_INITIAL_FFN')!='1'
                or os.environ.get('UNIFOLM_INITIAL_WORLD_LINEARS')!='1'
                or os.environ.get('UNIFOLM_DECISION_INITIAL_PRIMARY')!='ffn'):
            raise ValueError('World initial native hidden parts requires accepted initial stage wrappers')
        from .world_initial_native_hidden_parts import install_world_initial_native_hidden_parts
        initial_hidden_parts_count=install_world_initial_native_hidden_parts(model.model.diffusion_model)
        if initial_hidden_parts_count!=1:raise ValueError('Unexpected world initial FFN count')
    print(f'WORLD_INITIAL_NATIVE_HIDDEN_PARTS_MODEL mode={initial_hidden_parts} modules={initial_hidden_parts_count} '
          'original_BK32_native_GEGLU=true producer=64x64_w4_s2 final=64x128_w4_s3_three_products '
          'dynamic_hidden_parts=fresh_each_call static_weights=first_timed_warm_version_refresh_same_storage '
          'master_public_output=fp32 decision=previous',flush=True)
'''+anchor)
ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_INITIAL_BK32_NATIVE_GEGLU=${UNIFOLM_INITIAL_BK32_NATIVE_GEGLU:-0}"\n'
path.write_text(replace(source,anchor,anchor+'    echo "UNIFOLM_WORLD_INITIAL_NATIVE_HIDDEN_PARTS=${UNIFOLM_WORLD_INITIAL_NATIVE_HIDDEN_PARTS:-0}"\n'))
source=(root/'scripts/profile_world_geglu_hidden_parts_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS','UNIFOLM_WORLD_INITIAL_NATIVE_HIDDEN_PARTS')
source=source.replace('world_geglu_hidden_parts as fusion_helper','world_initial_native_hidden_parts as fusion_helper')
source=source.replace('fusion_helper.packed_hidden_ffn_residual','fusion_helper.initial_packed_final')
source=replace(source,"width=str(args[1].shape[-1])", "width=str(args[0][0].shape[0])")
source=source.replace('install_world_geglu_hidden_parts(model.model.diffusion_model)==10','install_world_initial_native_hidden_parts(model.model.diffusion_model)==1')
source=source.replace('_world_geglu_hidden_parts_installed','_world_initial_native_hidden_parts_installed')
source=source.replace('_world_geglu_hidden_parts_calls','_world_initial_native_hidden_parts_calls')
source=replace(source,'assert len(tail_dispatch)==10','assert len(tail_dispatch)==1')
source=replace(source,'fused_calls.get(str(width),0)>=30 for width in [640]','fused_calls.get(str(width),0)>=3 for width in [512]')
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','WORLD_INITIAL_NATIVE_HIDDEN_PARTS_MODEL')
ast.parse(source);(root/'scripts/profile_world_initial_native_hidden_parts_pair.py').write_text(source)
source=(root/'scripts/run_world_geglu_hidden_parts_pair.py').read_text()
source=replace(source,"flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='0'", "flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='1'\nflags['UNIFOLM_WORLD_INITIAL_NATIVE_HIDDEN_PARTS']='0'")
source=source.replace('world-geglu-hidden-parts-graph-pair-','world-initial-native-hidden-parts-graph-pair-')
source=source.replace('scripts/profile_world_geglu_hidden_parts_pair.py','scripts/profile_world_initial_native_hidden_parts_pair.py')
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','WORLD_INITIAL_NATIVE_HIDDEN_PARTS_MODEL')
ast.parse(source);(root/'scripts/run_world_initial_native_hidden_parts_pair.py').write_text(source)
for prefix in ['wait','run']:
 source=(root/f'scripts/{prefix}_world_geglu_hidden_parts_reserved_pair.sh').read_text()
 source=source.replace('world_geglu_hidden_parts','world_initial_native_hidden_parts')
 source=source.replace('world-geglu-hidden-parts-pair-','world-initial-native-hidden-parts-pair-')
 source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','WORLD_INITIAL_NATIVE_HIDDEN_PARTS_MODEL')
 (root/f'scripts/{prefix}_world_initial_native_hidden_parts_reserved_pair.sh').write_text(source)
print('WORLD_INITIAL_NATIVE_HIDDEN_PARTS_PREPARED original_BK32_native_math=true modules=1 decision_unchanged=true')
