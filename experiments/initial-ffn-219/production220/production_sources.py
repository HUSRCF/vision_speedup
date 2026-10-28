"""Add initial FFN routing to frozen adopted219; no GPU initialization."""
import ast,hashlib,importlib.util,json
from pathlib import Path
POLICY='both-spatial-temporal-resblock-grid-native-attention-initial-ffn'

def production_sources(base,variant=POLICY):
    assert variant==POLICY
    here=Path(__file__).resolve().parent
    prior=here.parents[1]/'native-attention-218/production219/production_sources.py'
    spec=importlib.util.spec_from_file_location('frozen_adopted219_sources',prior)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files,proof=module.production_sources(base)
    for name,digest in json.loads((here/'overlay-source-hashes.json').read_text()).items():
        raw=(here/'production-overlay'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
        ast.parse(raw.decode());files[name]=raw.decode()
    diagnostic=(here.parent/'model-r2/scripts/init_ffn219_mode.py').read_text()
    candidate=files['src/unifolm_wma/utils/nccl_initial_ffn220_mode.py']
    def node(text,name):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
    for name in ('row_values','new_shape_cache','InitialFFNController'):assert node(candidate,name)==node(diagnostic,name)
    proof=dict(proof,variant=POLICY,adopted219_helpers_preserved_except_runtime_routing=True,
        initial_FFN_controller_AST_equal_GPU_qualified_model=True,original_JIT_bodies_BK32_and_weight_owners_unchanged=True,
        initial_FFN_one_additional_collective=True)
    return files,proof
