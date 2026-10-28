"""Frozen temporal221 candidate on adopted220; imports no GPU libraries."""
import ast,hashlib,importlib.util,json
from pathlib import Path
POLICY='both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block'

def production_sources(base,variant=POLICY):
    assert variant==POLICY
    here=Path(__file__).resolve().parent
    prior=here.parents[1]/'initial-ffn-219/production220/production_sources.py'
    spec=importlib.util.spec_from_file_location('frozen_adopted220_sources',prior)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files,proof=module.production_sources(base)
    for name,digest in json.loads((here/'overlay-source-hashes.json').read_text()).items():
        raw=(here/'production-overlay'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
        ast.parse(raw.decode());files[name]=raw.decode()
    diagnostic=(here.parent/'model-r2/scripts/temporal220_block_mode.py').read_text()
    candidate=files['src/unifolm_wma/utils/nccl_temporal_block221_mode.py']
    def node(text,name):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
    for name in ('row_view','grouped_route','dispatch','new_shape_cache','launch','install','uninstall','mode'):assert node(candidate,name)==node(diagnostic,name)
    def controller(text):
        tree=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='Controller')
        tree.body=[n for n in tree.body if getattr(n,'name',None)!='report']
        return ast.dump(tree,include_attributes=False)
    assert controller(candidate)==controller(diagnostic)
    proof=dict(proof,variant=POLICY,adopted220_helpers_preserved_except_runtime_routing=True,
      temporal_routing_AST_equal_six_state_GPU_model=True,temporal_Controller_AST_equal_except_report=True,
      original209_and110JIT_bodies_unchanged=True,no_new_collectives=True,
      full5D_GroupNorm_all16_time_steps_and_init_attn220_preserved=True,
      grouped_scratch_and_copy_and_all_setup_inside_process_timer=True)
    return files,proof
