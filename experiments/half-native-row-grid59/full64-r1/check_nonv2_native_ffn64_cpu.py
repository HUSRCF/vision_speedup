"""Exercise two independent ownership batches and partial-bind teardown."""
import ast,types
from pathlib import Path
from build_nonv2_native_ffn64_sources import ADAPTER,production_sources
from experiment_remote import WORKSPACE

class Parent:
    def bind(self,graph):
        assert len(self.rows)==31 and len(self.records)==62 and not self.owners
        self.owners.extend(graph)
        if len(graph)!=62:raise ValueError('fixture partial binding')
        self.binding_reports.append(dict(kernels=list(graph)))

ns={'Controller':Parent};exec(ADAPTER,ns);C=ns['ProductionController']
def fixture():return C()
x=fixture();x.rows=list(range(31));x.records=list(range(62));x.owners=[];x.binding_reports=[]
x.bind(list(range(62)));x.rows.extend(range(31));x.records.extend(range(62));x.bind(list(range(62,124)))
assert len(x.rows)==62 and len(x.records)==124 and x.owners==list(range(124)) and len(x.binding_reports)==2
x.rows.extend(range(31));x.records.extend(range(62))
try:x.bind([124]);raise AssertionError('expected partial fixture failure')
except ValueError:pass
assert len(x.rows)==93 and len(x.records)==186 and x.owners==list(range(125))
files,proof=production_sources(WORKSPACE/'vision_speedup/experiments/nonv2-vae-all-display/unifolm')
runtime=files['src/unifolm_wma/utils/nccl_nonv2_native_ffn_runtime64.py']
assert runtime.index('_controller.close_owners()')<runtime.index("entry['graph'].reset()")
module=ast.parse(runtime);imports=[n for n in module.body if isinstance(n,(ast.Import,ast.ImportFrom))]
assert not any('torch'in ast.unparse(n) for n in imports)
assert proof['all_bind_capture_and_extra_replay_inside_process_timer']
print('NONV2_NATIVE_FFN64_CPU passed=true two_graphs_and_partial_bind_cleanup=true no_GPU=true')
