"""Wire decision-only primary FFN residual fusion and actual graph comparisons."""
import ast
from pathlib import Path
import sys
root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'


def replace(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


path=utils/'inference_runtime.py';source=path.read_text()
anchor='    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")\n'
source=replace(source,anchor,'''    decision_ffn_residual=os.environ.get('UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE','0')
    if decision_ffn_residual not in {'0','1'}:raise ValueError('Invalid decision FFN residual mode')
    decision_ffn_residual_count=0
    if decision_ffn_residual=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_FUSED_RESIDUAL_NORM')!='1' or
                os.environ.get('UNIFOLM_FFN_DTYPE')!='fp32' or
                os.environ.get('UNIFOLM_DECISION_PRIMARY_FFN')!='1280' or
                os.environ.get('UNIFOLM_DECISION_VENDOR_FFN')!='chunked'):
            raise ValueError('Decision FFN residual fusion requires accepted primary C1280 and FP32 norm policy')
        from .decision_ffn_residual_epilogue import install_decision_ffn_residual_epilogue
        decision_ffn_residual_count=install_decision_ffn_residual_epilogue(model.model.diffusion_model)
        if decision_ffn_residual_count!=12:raise ValueError('Unexpected decision C1280 residual block count')
    print(f'DECISION_FFN_RESIDUAL_EPILOGUE_MODEL mode={decision_ffn_residual} blocks={decision_ffn_residual_count} '
          'GEMM=previous_cublas_primary scale_bias_residual=fp32 '
          'static_pack=first_timed_warm_version_refresh_same_storage world=previous',flush=True)
'''+anchor)
ast.parse(source);path.write_text(source)
path=root/'scripts/team07_run.sh';source=path.read_text()
anchor='    echo "UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE=${UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE:-0}"\n'
path.write_text(replace(source,anchor,anchor+'    echo "UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE=${UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE:-0}"\n'))
source=(root/'scripts/profile_world_ffn640_residual_epilogue_pair.py').read_text()
source=source.replace('UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE','UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE')
source=source.replace('world_ffn640_residual_epilogue as fusion_helper','decision_ffn_residual_epilogue as fusion_helper')
source=source.replace('fusion_helper.ffn640_residual_epilogue','fusion_helper.primary_linear_residual')
source=replace(source,'                        assert fusion_helper.install_world_ffn640_residual_epilogue(model.model.diffusion_model)==10',
    '                        assert fusion_helper.install_decision_ffn_residual_epilogue(model.model.diffusion_model)==12')
start=source.index('                module_dispatch=');end=source.index('                for _ in range(3):',start)
source=source[:start]+'''                module_dispatch={name:getattr(m,'_decision_ffn_residual_epilogue_calls',0)
                    for name,m in model.model.diffusion_model.named_modules()
                    if getattr(m,'_decision_ffn_residual_epilogue_installed',False)}
                assert len(module_dispatch)==12
                if options.profile_stage=='decision':
                    assert all(n>=3 for n in module_dispatch.values()),module_dispatch
                    assert fused_calls.get('1280',0)>=36,fused_calls
                else:
                    assert not any(module_dispatch.values()),module_dispatch
                    assert not fused_calls,fused_calls
'''+source[end:]
source=replace(source,'fused_tail_calls_per_width=dict(fused_calls), tail_dispatch_per_module=tail_dispatch, norm3_dispatch_per_module=module_dispatch,',
    'fused_primary_calls_per_width=dict(fused_calls), decision_ffn_dispatch_per_module=module_dispatch,')
source=source.replace('WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL','DECISION_FFN_RESIDUAL_EPILOGUE_MODEL')
ast.parse(source);(root/'scripts/profile_decision_ffn_residual_epilogue_pair.py').write_text(source)
source=(root/'scripts/run_world_ffn640_residual_epilogue_pair.py').read_text()
source=replace(source,"flags['UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE']='0'",
    "flags['UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE']='1'\nflags['UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE']='0'")
source=source.replace('world-ffn640-residual-epilogue-graph-pair-','decision-ffn-residual-epilogue-graph-pair-')
source=source.replace('scripts/profile_world_ffn640_residual_epilogue_pair.py','scripts/profile_decision_ffn_residual_epilogue_pair.py')
source=source.replace('WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL','DECISION_FFN_RESIDUAL_EPILOGUE_MODEL')
ast.parse(source);(root/'scripts/run_decision_ffn_residual_epilogue_pair.py').write_text(source)
for prefix in ['wait','run']:
    source=(root/f'scripts/{prefix}_world_ffn640_residual_epilogue_reserved_pair.sh').read_text()
    source=source.replace('world_ffn640_residual_epilogue','decision_ffn_residual_epilogue')
    source=source.replace('world-ffn640-residual-epilogue-pair-','decision-ffn-residual-epilogue-pair-')
    source=source.replace('WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL','DECISION_FFN_RESIDUAL_EPILOGUE_MODEL')
    (root/f'scripts/{prefix}_decision_ffn_residual_epilogue_reserved_pair.sh').write_text(source)
print('DECISION_FFN_RESIDUAL_EPILOGUE_PREPARED blocks=12 primary_GEMM=original residual=finish_only world=previous',flush=True)
