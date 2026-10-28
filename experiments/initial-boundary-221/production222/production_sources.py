"""Frozen initial-block222 on adopted221; no GPU libraries on import."""
import ast,hashlib,importlib.util,json
from pathlib import Path
POLICY='both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block'

def production_sources(base,variant=POLICY):
    assert variant==POLICY
    here=Path(__file__).resolve().parent
    prior=here.parents[1]/'temporal-block-220/production221/production_sources.py'
    spec=importlib.util.spec_from_file_location('adopted221_for_boundary222',prior)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files,previous=module.production_sources(base)
    for name,digest in json.loads((here/'overlay-source-hashes.json').read_text()).items():
        raw=(here/'production-overlay'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
        ast.parse(raw.decode());files[name]=raw.decode()
    diagnostic=(here.parent/'world-model-r3/scripts/initial_boundary221_mode.py').read_text()
    helper=files['src/unifolm_wma/utils/nccl_initial_block222_mode.py']
    def node(text,name,omit_report=False):
        item=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        if omit_report:item.body=[n for n in item.body if getattr(n,'name',None)!='report']
        return ast.dump(item,include_attributes=False)
    for name in ['DeferredInitialGather','launch','install','uninstall','mode']:assert node(helper,name)==node(diagnostic,name)
    assert node(helper,'Controller',True)==node(diagnostic,'Controller',True)
    proof=dict(previous,variant=POLICY,initial_boundary_routing_AST_equal_world_model_r3=True,
        initial_boundary_Controller_AST_equal_except_report=True,original209_and110JIT_bodies_unchanged=True,
        full5D_GroupNorm_all16_time_steps_and_init_attn220_preserved=False,
        full5D_GroupNorm_all16_time_steps_and_Conv1d_projections_preserved=True,
        initial_FFN_BK32_arithmetic_and_version_owners_preserved=True,
        existing_initial_gather_moved_after_residual=True,initial_QKV_copy_nodes_included=2,no_new_collectives=True)
    assert proof==json.loads((here/'timing-proof.json').read_text())
    return files,proof
