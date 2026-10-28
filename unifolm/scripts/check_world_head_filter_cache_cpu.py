import json
from pathlib import Path
import torch
from unifolm_wma.utils import world_head_filter_cache as r
assert not torch.cuda.is_initialized()
env={r.FLAG:'1','UNIFOLM_CASE':'unitree_z1_dual_arm_stackbox_v2',**r.DEPENDENCIES}
assert r.configured(env) and not r.configured({})
for raw in ('bad','true','2'):
 try:r.configured({r.FLAG:raw})
 except ValueError:pass
 else:raise AssertionError('invalid flag')
assert not r.configured({**env,'UNIFOLM_CASE':'other'})
for key in r.DEPENDENCIES:
 bad=dict(env);bad.pop(key)
 try:r.configured(bad)
 except ValueError:pass
 else:raise AssertionError(key)
assert not torch.cuda.is_initialized()
report=dict(passed=True,CUDA_context_initialized=False,invalid_flags_rejected=True,
 dependency_guards=len(r.DEPENDENCIES),V2_only=True,default_disabled=True)
Path('world-head-filter-cache-production-cpu.json').write_text(json.dumps(report,indent=2)+chr(10))
print(json.dumps(report))
