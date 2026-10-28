"""Add an opt-in installer and independent runtime/actual-model checks."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1])
path=root/'src/unifolm_wma/utils/inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
assert source.count(anchor)==1 and 'UNIFOLM_DECISION_QKV_PACKED_ATTENTION' not in source
addition='''    decision_qkv_attention=os.environ.get('UNIFOLM_DECISION_QKV_PACKED_ATTENTION','0')
    if decision_qkv_attention not in {'0','1'}:
        raise ValueError('Invalid decision QKV packed attention mode')
    decision_qkv_attention_count=0
    if decision_qkv_attention=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_DECISION_PRIMARY_GROUPED_QKV')!='1' or
                os.environ.get('UNIFOLM_DECISION_PRIMARY_GROUPED_QKV_SCOPE')!='all' or
                os.environ.get('UNIFOLM_QKV_PACKED_ATTENTION')!='all' or
                os.environ.get('UNIFOLM_COMPENSATED_SPATIAL_ATTENTION')!='large' or
                os.environ.get('UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV')!='kv'):
            raise ValueError('Decision QKV fusion requires accepted V2 primary QKV and attention')
        from .decision_qkv_packed_attention import install_decision_qkv_packed_attention
        decision_qkv_attention_count=install_decision_qkv_packed_attention(model.model.diffusion_model)
        if decision_qkv_attention_count!=5:
            raise ValueError('Unexpected decision C320 spatial QKV count')
    print(f'DECISION_QKV_PACKED_ATTENTION_MODEL enabled={decision_qkv_attention} '
          f'modules={decision_qkv_attention_count} primary_BM128_BN128_BK64_group32_stages3=true '
          'attention=previous_BM64_BN64 master_public_output=fp32 input_conversion=each_call '
          'static_weights=first_timed_warm cache=previous_primary_QKV_same_storage world=previous',flush=True)
'''
source=source.replace(anchor,addition+anchor);ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_QKV_PACKED_ATTENTION=${UNIFOLM_QKV_PACKED_ATTENTION:-0}"\n'
assert source.count(anchor)==1
path.write_text(source.replace(anchor,anchor+'    echo "UNIFOLM_DECISION_QKV_PACKED_ATTENTION=${UNIFOLM_DECISION_QKV_PACKED_ATTENTION:-0}"\n'))

# Derive checks from the already accepted world-fusion check; no old source is
# replaced. The fallback fixture remains explicitly a dispatch sentinel.
source=(root/'scripts/check_world_qkv_packed_attention.py').read_text()
source=source.replace('world_qkv_packed_attention','decision_qkv_packed_attention')
source=source.replace('_world_zero_weight_low_qkv_installed','_decision_primary_grouped_qkv_installed')
source=source.replace('_world_zero_weight_low_qkv_cache','_primary_grouped_qkv_cache')
source=source.replace('from unifolm_wma.utils.triton_zero_weight_low_linear import zero_weight_low_linear,weight_low_count',
                      'from unifolm_wma.utils.triton_primary_grouped_qkv import primary_grouped_qkv')
source=source.replace('parts=pack_half_weight(combined);count=weight_low_count(parts)', 'parts=pack_half_weight(combined)')
source=source.replace('zero_weight_low_linear(value,parts,count,output_groups=3)',
                      'primary_grouped_qkv(value,parts[0],tile=(128,128,32,4,3))')
source=source.replace('torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False',
                      'torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False')
source=source.replace('[(320,2560),(640,640)]','[(320,2560)]')
source=source.replace(',cache[3].data_ptr()','')
source=source.replace('                    assert (int(cache[3])==0)==quantized\n','')
source=source.replace("helper,'qkv_packed_attention'","helper,'primary_qkv_packed_attention'")
source=source.replace('target.allow_tf32=True;exact(forward(value),old(value));target.allow_tf32=False',
                      "previous=target.allow_tf32;target.allow_tf32=(flag!='cudnn');exact(forward(value),old(value));target.allow_tf32=previous")
source=source.replace('zero_nonzero_zero_predicate_graph=True,','')
source=source.replace('helper.install_decision_qkv_packed_attention(tree)==2','helper.install_decision_qkv_packed_attention(tree)==1')
source=source.replace("'_qkv_packed_attention_installed'","'_decision_qkv_packed_attention_installed'")
source=source.replace('QKV_PACKED_ATTENTION_RUNTIME','DECISION_QKV_PACKED_ATTENTION_RUNTIME')
source=source.replace('fixtures=4','fixtures=2')
assert 'cache[3]' not in source and 'zero_weight_low_linear(' not in source
ast.parse(source);(root/'scripts/check_decision_qkv_packed_attention.py').write_text(source)

source=(root/'scripts/profile_qkv_packed_attention_pair.py').read_text()
source=source.replace('UNIFOLM_QKV_PACKED_ATTENTION','UNIFOLM_DECISION_QKV_PACKED_ATTENTION')
source=source.replace('QKV_PACKED_ATTENTION_MODEL','DECISION_QKV_PACKED_ATTENTION_MODEL')
source=source.replace('world_qkv_packed_attention as fusion_helper','decision_qkv_packed_attention as fusion_helper')
source=source.replace('fusion_helper.qkv_packed_attention','fusion_helper.primary_qkv_packed_attention')
source=source.replace("('candidate','all')","('candidate','1')")
source=source.replace('install_world_qkv_packed_attention(model.model.diffusion_model)==10',
                      'install_decision_qkv_packed_attention(model.model.diffusion_model)==5')
source=source.replace("if options.profile_stage=='world':","if options.profile_stage=='decision':")
source=source.replace('for width in [320,640]','for width in [320]')
ast.parse(source);(root/'scripts/profile_decision_qkv_packed_attention_pair.py').write_text(source)
source=(root/'scripts/run_qkv_packed_attention_pair.py').read_text()
source=source.replace('UNIFOLM_QKV_PACKED_ATTENTION','UNIFOLM_DECISION_QKV_PACKED_ATTENTION')
source=source.replace('qkv-packed-attention-graph-pair-','decision-qkv-packed-attention-graph-pair-')
source=source.replace('scripts/profile_qkv_packed_attention_pair.py','scripts/profile_decision_qkv_packed_attention_pair.py')
source=source.replace('QKV_PACKED_ATTENTION_MODEL','DECISION_QKV_PACKED_ATTENTION_MODEL')
anchor="flags['UNIFOLM_DECISION_QKV_PACKED_ATTENTION']='0'\n"
assert source.count(anchor)==1
source=source.replace(anchor,anchor+"assert flags['UNIFOLM_QKV_PACKED_ATTENTION']=='all'\n")
ast.parse(source);(root/'scripts/run_decision_qkv_packed_attention_pair.py').write_text(source)
source=(root/'scripts/wait_qkv_packed_attention_reserved_pair.sh').read_text()
source=source.replace('run_qkv_packed_attention_reserved_pair.sh','run_decision_qkv_packed_attention_reserved_pair.sh')
source=source.replace('qkv-packed-attention-pair-','decision-qkv-packed-attention-pair-')
(root/'scripts/wait_decision_qkv_packed_attention_reserved_pair.sh').write_text(source)
source=(root/'scripts/run_qkv_packed_attention_reserved_pair.sh').read_text()
source=source.replace('run_qkv_packed_attention_pair.py','run_decision_qkv_packed_attention_pair.py')
source=source.replace('QKV_PACKED_ATTENTION_MODEL_BOTH_STAGES_CHECK','DECISION_QKV_PACKED_ATTENTION_MODEL_BOTH_STAGES_CHECK')
(root/'scripts/run_decision_qkv_packed_attention_reserved_pair.sh').write_text(source)
print('DECISION_QKV_PACKED_ATTENTION_PREPARED primary_dot_geometry_preserved=true '
      'bothstage_signedzero_and_decision_dispatch_counts_required=true',flush=True)
