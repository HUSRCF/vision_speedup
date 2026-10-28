"""Build the opt-in storage adapter and check CPU mmap/isolation, without CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workdir',type=Path,required=True,help='Fresh private directory on tmpfs, e.g. /var/tmp/team07-storage-test')
    parser.add_argument('--exec-dir',type=Path,required=True,help='Fresh private directory on an executable filesystem')
    args=parser.parse_args();here=Path(__file__).resolve().parent
    work=args.workdir.resolve();executable=args.exec_dir.resolve()
    assert not work.exists()and not executable.exists()
    work.mkdir(parents=True,mode=0o700);executable.mkdir(parents=True,mode=0o700)
    shared=work/'shared';shared.mkdir(mode=0o700)
    source=here/'cpu-r1/nccl_private_tmpfs.c';library=executable/'libteam07_nccl_private_tmpfs.so'
    qualified=json.loads((here/'qualified-adapter.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==qualified['source_sha256']
    env=dict(os.environ,TMPDIR=str(work),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['cc','-shared','-fPIC','-O2','-Wall','-Wextra','-Werror','-o',str(library),str(source),'-ldl'],env=env,check=True)
    env.update(LD_PRELOAD=str(library),TEAM07_NCCL_SHM_ROOT=str(shared),CPU_STORAGE_REPORT=str(work/'cpu-report.json'))
    subprocess.run([sys.executable,str(here/'cpu-r1/check_nccl_private_tmpfs.py')],env=env,check=True)
    report=json.loads((work/'cpu-report.json').read_text());assert report['passed']and report['no_GPU_execution']
    config=dict(source_sha256=qualified['source_sha256'],preload=str(library),preload_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),shared_root=str(shared),cpu_cross_process_passed=True,no_GPU_execution=True)
    path=work/'storage-config.json';path.write_text(json.dumps(config,indent=2)+'\n')
    print(path)


if __name__=='__main__':main()
