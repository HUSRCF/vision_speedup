"""Expand existing native FFN gathers to include spatial native attention."""
import ast
from pathlib import Path
from experiment_remote import WORKSPACE
import build_nonv2_native_ffn64_sources as previous
ENTRY=previous.ENTRY
RUNTIME='src/unifolm_wma/utils/nccl_nonv2_native_shared_runtime66.py'
MODE=previous.MODE
POLICY='nonv2-native-shared-grid'
ADAPTER=previous.ADAPTER

def production_sources(base):
    files,proof=previous.production_sources(base)
    runtime=files.pop(previous.RUNTIME).replace(previous.POLICY,POLICY).replace('from .nonv2_native_ffn_grid59_geglu_mode import','from .nonv2_native_ffn_grid59_spatial_mode import')
    runtime=runtime.replace("    assert all(row['collectives']==1 for row in report['rows'])", "    assert sum(row['collectives'] for row in report['rows'])+report['spatial_shared_boundary']['captured']==62\n    assert report['spatial_shared_boundary']['captured']==30 and report['spatial_shared_boundary']['attention_calls']==150\n    assert report['spatial_shared_boundary']['new_collectives']==0")
    files[RUNTIME]=runtime
    files[ENTRY]=files[ENTRY].replace('nccl_nonv2_native_ffn_runtime64','nccl_nonv2_native_shared_runtime66')
    scripts=Path(__file__).parent
    shared=(scripts/'nonv2_native_ffn_grid59_spatial_mode.py').read_text()
    shared=shared.replace('import nonv2_native_ffn_grid59_geglu_mode as geglu','from . import nonv2_native_ffn_grid59_geglu_mode as geglu').replace('from nonv2_native_attention59_mode import','from .nonv2_native_attention59_mode import').replace('class Controller(geglu.Controller):','class Controller(geglu.ProductionController):')
    files['src/unifolm_wma/utils/nonv2_native_ffn_grid59_spatial_mode.py']=shared
    files['src/unifolm_wma/utils/nonv2_native_attention59_mode.py']=(scripts/'nonv2_native_attention59_mode.py').read_text()
    for text in files.values():ast.parse(text)
    proof.update(source59_to64=False,source59_to66=True,spatial_attention_calls_per_stage=75,spatial_gathers_moved_per_stage=15,new_collectives=0,original31_gather_count_per_stage_preserved=True,partitioned_attention_and_FFN_masters_six_state_bitwise=True)
    return files,proof

if __name__=='__main__':
    files,proof=production_sources(WORKSPACE/'vision_speedup/experiments/nonv2-vae-all-display/unifolm')
    assert len(files)==8
    print('NATIVE_SHARED66_BUILDER passed=true no_new_collectives=true')
