"""Exercise storage rejection and lock release before any CUDA allocation."""
import argparse,ast,hashlib,json,os,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();a.source=a.source.resolve()
    here=Path(__file__).resolve().parent
    tree=ast.parse((here/'wait_nccl_full209.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    code=compile(ast.Module(body=[function],type_ignores=[]),'<guard-main-fixture>','exec')
    cases=[]
    class AllocationReached(Exception):pass
    with tempfile.TemporaryDirectory(prefix='initial222-guard-cpu-')as directory:
        work=Path(directory);(work/'flags.json').write_bytes((here/'flags.json').read_bytes())
        before=Path.cwd();os.chdir(work)
        try:
            for label,ram,tmp,reaches in [('low_RAM',23*2**20,99*2**20,False),('low_tmp',99*2**20,10*2**20,False),('enough',99*2**20,99*2**20,True)]:
                calls=[];closed=[]
                def empty(*args,**kwargs):calls.append(1);raise AllocationReached
                def disk(path):return SimpleNamespace(free=tmp if str(path)=='/tmp'else ram)
                lock=SimpleNamespace(close=lambda:closed.append(1))
                ns=dict(Path=Path,os=os,hashlib=hashlib,json=json,
                    shutil=SimpleNamespace(disk_usage=disk),reserve=lambda:[lock],
                    torch=SimpleNamespace(empty=empty,uint8=object()))
                exec(code,ns)
                with patch.dict(os.environ,{'CANONICAL_ROOT':str(a.source.resolve())}):
                    try:ns['main']()
                    except AllocationReached:assert reaches
                    except AssertionError:assert not reaches
                    else:raise AssertionError('fixture failed to stop before GPU work')
                assert bool(calls)==reaches and closed==[1]
                cases.append(dict(case=label,allocation_attempted=reaches,lock_released=True))
        finally:os.chdir(before)
    print(json.dumps(dict(passed=True,cases=cases,no_Torch_or_CUDA_import=True)))


if __name__=='__main__':main()
