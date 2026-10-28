"""Check the exact timed overlay and routing source without touching a GPU."""
import ast,hashlib,json,tempfile,subprocess,sys
from pathlib import Path
from production_sources import production_sources
here=Path(__file__).resolve().parent;base=here.parents[2]/'unifolm'
files,proof=production_sources(base)
old=json.loads((base/'world-head-filter-cache-production-sources.json').read_text())
manifest=dict(old);manifest.update({n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items() if n.startswith('src/')})
assert len(manifest)==219 and manifest==json.loads((here/'production-sources.json').read_text())
assert proof==json.loads((here/'timing-proof.json').read_text())
assert proof['four_additive_lifecycle_hooks'] and proof['removing_only_hooks_recovers_original_exact_bytes']
assert proof['original_timer_assignment_and_profile_writer_unchanged'] and proof['original_four_timing_hooks_unchanged']
new=files['src/unifolm_wma/utils/nccl_native_attention219_mode.py']
assert not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(ast.unparse(d).endswith('.jit') for d in n.decorator_list) for n in ast.walk(ast.parse(new)))
assert 'dist.all_gather' not in new and 'dist.all_reduce' not in new
print(json.dumps(dict(passed=True,fixed_sources=219,original209_unchanged=True,no_new_JIT_or_collective=True,exact_GPU_qualified_attention_AST=True,original_timer_and_four_hooks_unchanged=True,no_GPU_initialized=True)))
