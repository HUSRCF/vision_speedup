"""Install norm3 packing only under the precisely validated V2 world policy."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_WORLD_NORM3_PACKED_GEGLU' not in source
addition='''    norm3_pack_scope=os.environ.get('UNIFOLM_WORLD_NORM3_PACKED_GEGLU','0')
    if norm3_pack_scope not in {'0','all','c320','c640','c1280'}:
        raise ValueError('Invalid world norm3 GEGLU packing scope')
    norm3_pack_count=0
    if norm3_pack_scope!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_FUSED_RESIDUAL_NORM')!='1' or
                os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU')!='all' or
                os.environ.get('UNIFOLM_WORLD_C320_BK64_GEGLU')!='gate' or
                os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY')!='640' or
                os.environ.get('UNIFOLM_PREPACKED_GEGLU')!='1'):
            raise ValueError('Norm3 packing requires accepted residual norm and world GEGLU policy')
        from .world_norm3_packed_geglu import install_world_norm3_packed_geglu
        norm3_pack_count=install_world_norm3_packed_geglu(model.model.diffusion_model,norm3_pack_scope)
        if norm3_pack_count!={'all':32,'c320':10,'c640':10,'c1280':12}[norm3_pack_scope]:
            raise ValueError('Unexpected norm3 GEGLU block count')
    print(f'WORLD_NORM3_PACKED_GEGLU_MODEL scope={norm3_pack_scope} blocks={norm3_pack_count} '
          'norm_math=previous_reduction_layout4_half_store_address_identity '
          'GEGLU=previous_JIT_BK64_BM_BN_group_primary_modes master_public_output=fp32 '
          'dynamic_norm_packing=each_call weight_pack=first_timed_warm_version_refresh_same_storage '
          'decision=previous initial512=previous',flush=True)
'''
source=source.replace(anchor,addition+anchor);ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU=${UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU:-0}"\n'
assert source.count(anchor)==1
path.write_text(source.replace(anchor,anchor+'    echo "UNIFOLM_WORLD_NORM3_PACKED_GEGLU=${UNIFOLM_WORLD_NORM3_PACKED_GEGLU:-0}"\n'))
source=(root/'scripts/profile_decision_qkv_packed_attention_pair.py').read_text()
source=source.replace('UNIFOLM_DECISION_QKV_PACKED_ATTENTION','UNIFOLM_WORLD_NORM3_PACKED_GEGLU')
source=source.replace('DECISION_QKV_PACKED_ATTENTION_MODEL','WORLD_NORM3_PACKED_GEGLU_MODEL')
source=source.replace('decision_qkv_packed_attention as fusion_helper','world_norm3_packed_geglu as fusion_helper')
source=source.replace('fusion_helper.primary_qkv_packed_attention','fusion_helper.residual_norm_half_pack')
source=source.replace("('candidate','1')","('candidate','all')")
source=source.replace('install_decision_qkv_packed_attention(model.model.diffusion_model)==5',
                      'install_world_norm3_packed_geglu(model.model.diffusion_model)==32')
source=source.replace("if options.profile_stage=='decision':","if options.profile_stage=='world':")
source=source.replace("all(fused_calls.get(str(width),0)>=15 for width in [320])",
                      "all(fused_calls.get(str(width),0)>=30 for width in [320,640,1280])")
anchor="                if options.profile_stage=='world':\n"
assert source.count(anchor)==1
source=source.replace(anchor,'''                module_dispatch={name:getattr(m,'_world_norm3_packed_geglu_calls',0)
                    for name,m in model.model.diffusion_model.named_modules()
                    if getattr(m,'_world_norm3_packed_geglu_installed',False)}
                assert len(module_dispatch)==32
                if options.profile_stage=='world':
                    assert all(n>=3 for n in module_dispatch.values()),module_dispatch
                else:
                    assert not any(module_dispatch.values()),module_dispatch
'''+anchor)
source=source.replace('fused_qkv_python_calls_per_width=dict(fused_calls),',
                      'fused_norm3_calls_per_width=dict(fused_calls), norm3_dispatch_per_module=module_dispatch,')
ast.parse(source);(root/'scripts/profile_world_norm3_packed_geglu_pair.py').write_text(source)
source=(root/'scripts/run_decision_qkv_packed_attention_pair.py').read_text()
source=source.replace('UNIFOLM_DECISION_QKV_PACKED_ATTENTION','UNIFOLM_WORLD_NORM3_PACKED_GEGLU')
source=source.replace('decision-qkv-packed-attention-graph-pair-','world-norm3-packed-geglu-graph-pair-')
source=source.replace('scripts/profile_decision_qkv_packed_attention_pair.py','scripts/profile_world_norm3_packed_geglu_pair.py')
source=source.replace('DECISION_QKV_PACKED_ATTENTION_MODEL','WORLD_NORM3_PACKED_GEGLU_MODEL')
anchor="flags['UNIFOLM_WORLD_NORM3_PACKED_GEGLU']='0'\n"
assert source.count(anchor)==1
source=source.replace(anchor,anchor+"assert flags['UNIFOLM_DECISION_QKV_PACKED_ATTENTION']=='1'\n")
ast.parse(source);(root/'scripts/run_world_norm3_packed_geglu_pair.py').write_text(source)
source=(root/'scripts/wait_decision_qkv_packed_attention_reserved_pair.sh').read_text()
source=source.replace('run_decision_qkv_packed_attention_reserved_pair.sh','run_world_norm3_packed_geglu_reserved_pair.sh')
source=source.replace('decision-qkv-packed-attention-pair-','world-norm3-packed-geglu-pair-')
(root/'scripts/wait_world_norm3_packed_geglu_reserved_pair.sh').write_text(source)
source=(root/'scripts/run_decision_qkv_packed_attention_reserved_pair.sh').read_text()
source=source.replace('run_decision_qkv_packed_attention_pair.py','run_world_norm3_packed_geglu_pair.py')
source=source.replace('DECISION_QKV_PACKED_ATTENTION_MODEL_BOTH_STAGES_CHECK','WORLD_NORM3_PACKED_GEGLU_MODEL_BOTH_STAGES_CHECK')
(root/'scripts/run_world_norm3_packed_geglu_reserved_pair.sh').write_text(source)
print('WORLD_NORM3_PACKED_GEGLU_PREPARED original_GEGLU_JIT=true original_norm_reduction_layout=true '
      'world_only=true bothstage_bits_and_each32block_dispatch_required=true',flush=True)
