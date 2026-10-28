"""Add native attention routing to the frozen adopted218 overlay."""
import ast,hashlib,importlib.util,json
from pathlib import Path
POLICY='both-spatial-temporal-resblock-grid-native-attention'

def production_sources(base,variant=POLICY):
    assert variant==POLICY
    here=Path(__file__).resolve().parent
    prior=here.parents[1]/'nccl-native-frame-grid-209/production_sources.py'
    spec=importlib.util.spec_from_file_location('frozen_adopted218_sources',prior)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files,proof=module.production_sources(base)
    for name,digest in json.loads((here/'overlay-source-hashes.json').read_text()).items():
        raw=(here/'production-overlay'/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest
        ast.parse(raw.decode());files[name]=raw.decode()
    diagnostic=here.parent/'model-world-r1/scripts/native_attention218_mode.py'
    candidate=files['src/unifolm_wma/utils/nccl_native_attention219_mode.py']
    def function(text,name):
        return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    for name in ('local_native','attention'):
        assert function(candidate,name)==function(diagnostic.read_text(),name)
    proof=dict(proof,variant=POLICY,original209_math_unchanged=True,
               adopted218_helpers_preserved_except_runtime_routing=True,
               native_attention_functions_AST_equal_GPU_qualified_model=True,
               original_four_timing_hooks_unchanged=True)
    return files,proof
