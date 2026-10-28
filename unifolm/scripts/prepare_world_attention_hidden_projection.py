"""Preserve accepted QKV launch prefix, stream attention directly into output GEMM."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'


def replace(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


source=(utils/'triton_qkv_packed_attention.py').read_text();lines=source.splitlines(keepends=True)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='qkv_packed_attention')
host=''.join(lines[node.lineno-1:node.end_lineno])
host=replace(host,'def qkv_packed_attention(value,parts,count,*,scale=.125,debug=False):',
    'def qkv_attention_hidden_projection(value,parts,count,output_parts,output_bias=None,*,scale=.125):')
start=host.index("    output=empty_attention('spatial',shaped)")
host=host[:start]+'''    hidden=attention_hidden_parts(shaped,(kh,kl,vh,vl),mode='vector')
    return packed_attention_output(hidden,output_parts,output_bias)
'''
new=ast.parse(host).body[0]
assert [ast.dump(n,include_attributes=False) for n in node.body[:-4]]==[ast.dump(n,include_attributes=False) for n in new.body[:-2]]
source='''import torch
import triton
from .triton_half_compensated_linear import pack_half_weight
from .triton_qkv_packed_attention import _qkv_pack
from .gemm_output_allocation import empty as empty_gemm
from .half_pack_allocation import empty_like as empty_half
from .attention_hidden_parts_pipeline import attention_hidden_parts,packed_attention_output

'''+host
ast.parse(source);(utils/'qkv_attention_hidden_projection.py').write_text(source)
path=utils/'inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
source=replace(source,anchor,'''    attention_hidden_scope=os.environ.get('UNIFOLM_WORLD_ATTENTION_HIDDEN_PROJECTION','0')
    if attention_hidden_scope not in {'0','all','c320','c640'}:raise ValueError('Invalid world attention hidden scope')
    attention_hidden_count=0
    if attention_hidden_scope!='0':
        if (os.environ.get('UNIFOLM_QKV_PACKED_ATTENTION')!='all'
                or os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_OUTPUT')!='input'
                or 'output640' not in os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE','').split('_ffn')[0]
                or os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT')!='input'):
            raise ValueError('Attention hidden projection requires original QKV/attention and input-protected output policies')
        from .world_attention_hidden_projection import install_world_attention_hidden_projection
        attention_hidden_count=install_world_attention_hidden_projection(model.model.diffusion_model,attention_hidden_scope)
        if attention_hidden_count!=(10 if attention_hidden_scope=='all' else 5):raise ValueError('Unexpected attention hidden projection count')
    print(f'WORLD_ATTENTION_HIDDEN_PROJECTION_MODEL mode={attention_hidden_scope} modules={attention_hidden_count} '
          'QKV=previous_shared_cache attention=previous_BM64_BN64_FP32_online_softmax_vector_Half_output '
          'projection=previous_input_protected_two_products_BK64_BM64_BN64_group32 '
          'dynamic_parts=fresh_each_call static_output_weights=first_timed_warm_version_refresh_same_storage '
          'master_public_output=fp32 decision=previous',flush=True)
'''+anchor)
ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_QKV_PACKED_ATTENTION=${UNIFOLM_QKV_PACKED_ATTENTION:-0}"\n'
path.write_text(replace(source,anchor,anchor+'    echo "UNIFOLM_WORLD_ATTENTION_HIDDEN_PROJECTION=${UNIFOLM_WORLD_ATTENTION_HIDDEN_PROJECTION:-0}"\n'))
source=(root/'scripts/profile_world_geglu_hidden_parts_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS','UNIFOLM_WORLD_ATTENTION_HIDDEN_PROJECTION')
source=source.replace('world_geglu_hidden_parts as fusion_helper','world_attention_hidden_projection as fusion_helper')
source=source.replace('fusion_helper.packed_hidden_ffn_residual','fusion_helper.qkv_attention_hidden_projection')
source=replace(source,"width=str(args[1].shape[-1])", "width=str(value.shape[-1])")
source=replace(source,"[('baseline','0'),('candidate','1')]", "[('baseline','0'),('candidate','all')]")
source=source.replace('install_world_geglu_hidden_parts(', 'install_world_attention_hidden_projection(')
source=source.replace('_world_geglu_hidden_parts_installed','_world_attention_hidden_projection_installed')
source=source.replace('_world_geglu_hidden_parts_calls','_world_attention_hidden_projection_calls')
source=replace(source,'for width in [640]','for width in [320,640]')
source=replace(source,'>=30 for width', '>=15 for width')
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','WORLD_ATTENTION_HIDDEN_PROJECTION_MODEL')
ast.parse(source);(root/'scripts/profile_world_attention_hidden_projection_pair.py').write_text(source)
source=(root/'scripts/run_world_geglu_hidden_parts_pair.py').read_text()
source=replace(source,"flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='0'", "flags['UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS']='1'\nflags['UNIFOLM_WORLD_ATTENTION_HIDDEN_PROJECTION']='0'")
source=source.replace('world-geglu-hidden-parts-graph-pair-','world-attention-hidden-projection-graph-pair-')
source=source.replace('scripts/profile_world_geglu_hidden_parts_pair.py','scripts/profile_world_attention_hidden_projection_pair.py')
source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','WORLD_ATTENTION_HIDDEN_PROJECTION_MODEL')
ast.parse(source);(root/'scripts/run_world_attention_hidden_projection_pair.py').write_text(source)
for prefix in ['wait','run']:
 source=(root/f'scripts/{prefix}_world_geglu_hidden_parts_reserved_pair.sh').read_text()
 source=source.replace('world_geglu_hidden_parts','world_attention_hidden_projection')
 source=source.replace('world-geglu-hidden-parts-pair-','world-attention-hidden-projection-pair-')
 source=source.replace('WORLD_GEGLU_HIDDEN_PARTS_MODEL','WORLD_ATTENTION_HIDDEN_PROJECTION_MODEL')
 (root/f'scripts/{prefix}_world_attention_hidden_projection_reserved_pair.sh').write_text(source)
print('WORLD_ATTENTION_HIDDEN_PROJECTION_PREPARED QKV_prefix_AST=true attention_BM64_BN64_vector=true original_projection_math=true world_modules=10')
