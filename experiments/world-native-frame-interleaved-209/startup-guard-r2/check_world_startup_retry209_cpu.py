"""Execute the real retry script against isolated failed-startup fixtures.

No Torch import or GPU allocation. An unverified context attempt must be
discarded, whereas a model numerical failure must stop without retrying.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',type=Path)
    args=parser.parse_args()
    if args.source_dir:
        files={name:(args.source_dir/name).read_text() for name in
               ('check_nccl_geglu_model_rank.py','retry_nccl_model_pair.py')}
    else:
        from probe_world_interleaved_model209 import sources
        files=sources()
    worker=files['check_nccl_geglu_model_rank.py']
    tree=ast.parse(worker)
    ownership=next(n for n in tree.body if isinstance(n,ast.Assert) and
                   ast.unparse(n.test)=='len(own) == 1')
    guard=ast.Expression(ownership.test)
    assert [eval(compile(guard,'ownership','eval'),{'own':set(range(i))}) for i in range(3)]==[False,True,False]
    fixtures=[('missing_rank_context', 'AssertionError: rank CUDA context ownership unresolved: nonparent=[]', True),
              ('extra_context', 'AssertionError: rank CUDA context ownership unresolved: nonparent=[1,2]', True),
              ('foreign_context', 'AssertionError: foreign GPU context before ranks start', True),
              ('model_numerics', 'AssertionError: video signedzero_bits mismatch', False)]
    report=[]
    for label,reason,retry in fixtures:
        with tempfile.TemporaryDirectory(prefix='world-startup-retry-cpu-') as name:
            root=Path(name);(root/'scripts').mkdir()
            (root/'flags.json').write_text('{}\n')
            (root/'check_nccl_geglu_model_rank.py').write_text('# fixture placeholder\n')
            (root/'retry_nccl_model_pair.py').write_text(files['retry_nccl_model_pair.py'])
            fake=f'''from pathlib import Path
import sys
r=Path.cwd()
if r.name=='attempt-1':
 (r/'ranks.log').write_text({reason!r}+'\\n')
 print('Parent waiting for ready files failed',flush=True)
 sys.exit(1)
for n in ('rank0-ready.json','rank1-ready.json','rank0-report.json','rank1-report.json','nccl-geglu-model-report.json'):
 (r/n).write_text('{{}}\\n')
(r/'ranks.log').write_text('fixture completed\\n')
'''
            (root/'wait_nccl_geglu_model_pair.py').write_text(fake)
            result=subprocess.run([sys.executable,str(root/'retry_nccl_model_pair.py')],cwd=root,
                                  capture_output=True,text=True,timeout=20)
            attempts=sorted(p.name for p in root.glob('attempt-*'))
            if retry:
                assert result.returncode==0 and attempts==['attempt-1','attempt-2'],result.stdout+result.stderr
                assert 'CONTAMINATED_ATTEMPT_EXCLUDED=1' in result.stdout
            else:
                assert result.returncode!=0 and attempts==['attempt-1'],result.stdout+result.stderr
                assert 'CONTAMINATED_ATTEMPT_EXCLUDED' not in result.stdout
            report.append(dict(fixture=label,passed=True,attempts=attempts,retried=retry))
    result=dict(passed=True,CPU_only=True,no_Torch_import=True,strict_exactly_one_rank_context_preserved=True,
                fixtures=report,retry_SHA256=hashlib.sha256(files['retry_nccl_model_pair.py'].encode()).hexdigest(),
                worker_SHA256=hashlib.sha256(worker.encode()).hexdigest(),GPU_model_result=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
