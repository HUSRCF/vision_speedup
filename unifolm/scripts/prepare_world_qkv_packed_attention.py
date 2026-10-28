"""Opt in to exact spatial QKV/attention fusion and build measured graph pairs."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_QKV_PACKED_ATTENTION' not in source
addition='''    qkv_attention_scope=os.environ.get('UNIFOLM_QKV_PACKED_ATTENTION','0')
    if qkv_attention_scope not in {'0','all','c320','c640'}:
        raise ValueError('Invalid QKV packed attention scope')
    qkv_attention_count=0
    if qkv_attention_scope!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or
                os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV')!='all' or
                os.environ.get('UNIFOLM_COMPENSATED_SPATIAL_ATTENTION')!='large' or
                os.environ.get('UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV')!='kv'):
            raise ValueError('QKV attention fusion requires accepted V2 world QKV/attention policy')
        from .world_qkv_packed_attention import install_world_qkv_packed_attention
        qkv_attention_count=install_world_qkv_packed_attention(model.model.diffusion_model,qkv_attention_scope)
        if qkv_attention_count!=(10 if qkv_attention_scope=='all' else 5):
            raise ValueError('Unexpected spatial QKV attention module count')
    print(f'QKV_PACKED_ATTENTION_MODEL scope={qkv_attention_scope} modules={qkv_attention_count} '
          'QKV=previous_BK64_group32_device_residual_predicate '
          'K_V_split=fused_previous_math attention=previous_BM64_BN64 '
          'master_public_output=fp32 input_pack=each_call static_weights=first_timed_warm '
          'cache=previous_shared_QKV_same_storage decision=previous',flush=True)
'''
source=source.replace(anchor,addition+anchor);ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV=${UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV:-0}"\n'
assert source.count(anchor)==1
path.write_text(source.replace(anchor,anchor+'    echo "UNIFOLM_QKV_PACKED_ATTENTION=${UNIFOLM_QKV_PACKED_ATTENTION:-0}"\n'))
source=(root/'scripts/profile_zero_weight_low_geglu_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU','UNIFOLM_QKV_PACKED_ATTENTION')
source=source.replace('ZERO_WEIGHT_LOW_GEGLU','QKV_PACKED_ATTENTION_MODEL')
anchor="                    os.environ['UNIFOLM_QKV_PACKED_ATTENTION'] = mode\n"
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'''                    if mode!='0':
                        assert fusion_helper.install_world_qkv_packed_attention(model.model.diffusion_model)==10
''')
anchor='            graphs = []\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'''            from unifolm_wma.utils import world_qkv_packed_attention as fusion_helper
            original_fused=fusion_helper.qkv_packed_attention
            fused_calls={}
            def counted(value,*args,**kwargs):
                width=str(value.shape[-1]);fused_calls[width]=fused_calls.get(width,0)+1
                return original_fused(value,*args,**kwargs)
            fusion_helper.qkv_packed_attention=counted
''')
anchor='                assert tensor_count > 0\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'''                if options.profile_stage=='world':
                    assert all(fused_calls.get(str(width),0)>=15 for width in [320,640]),fused_calls
                else:
                    assert sum(fused_calls.values())==0,fused_calls
''')
anchor='same_process=True, same_gpu=True, same_model_weights=True,'
assert source.count(anchor)==1
source=source.replace(anchor,'fused_qkv_python_calls_per_width=dict(fused_calls), '+anchor)
anchor='            finally:\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'                fusion_helper.qkv_packed_attention=original_fused\n')
ast.parse(source);(root/'scripts/profile_qkv_packed_attention_pair.py').write_text(source)
source=(root/'scripts/run_zero_weight_low_geglu_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU','UNIFOLM_QKV_PACKED_ATTENTION')
anchor="flags['UNIFOLM_WORLD_ZERO_WEIGHT_LOW']='0'\n"
assert source.count(anchor)==1;source=source.replace(anchor,'')
source=source.replace('zero-weight-low-geglu-graph-pair-','qkv-packed-attention-graph-pair-')
source=source.replace('scripts/profile_zero_weight_low_geglu_pair.py','scripts/profile_qkv_packed_attention_pair.py')
source=source.replace('ZERO_WEIGHT_LOW_GEGLU','QKV_PACKED_ATTENTION_MODEL')
ast.parse(source);(root/'scripts/run_qkv_packed_attention_pair.py').write_text(source)
source=(root/'scripts/wait_initial_native_geglu_reserved_pair.sh').read_text()
source=source.replace('run_initial_native_geglu_reserved_pair.sh','run_qkv_packed_attention_reserved_pair.sh')
source=source.replace('initial-native-pair-','qkv-packed-attention-pair-')
(root/'scripts/wait_qkv_packed_attention_reserved_pair.sh').write_text(source)
source=(root/'scripts/run_initial_native_geglu_reserved_pair.sh').read_text()
source=source.replace('run_initial_bk32_native_geglu_pair.py','run_qkv_packed_attention_pair.py')
source=source.replace('INITIAL_BK32_NATIVE_GEGLU_BOTH_STAGES_CHECK','QKV_PACKED_ATTENTION_MODEL_BOTH_STAGES_CHECK')
(root/'scripts/run_qkv_packed_attention_reserved_pair.sh').write_text(source)
print('WORLD_QKV_PACKED_ATTENTION_PREPARED spatial_only=true previous_QKV_cache=true '
      'same_attention_kernel=true bothstage_signedzero_and_world_dispatch_counts_required=true',flush=True)
