"""Run the source-frozen communication-only screen; never loads the model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scope',choices=['micro','combos'],default='micro')
    parser.add_argument('--gpus',required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--storage-config',type=Path,required=True)
    args=parser.parse_args();devices=args.gpus.split(',');assert len(devices)==len(set(devices))==2 and set(devices)<=set('0123')
    here=Path(__file__).resolve().parent;frozen=here/(args.scope+'-r1')
    config=json.loads(args.storage_config.read_text());assert config['cpu_cross_process_passed']
    qualified=json.loads((here.parent/'private-nccl-tmpfs/qualified-adapter.json').read_text())
    assert config['source_sha256']==qualified['source_sha256']
    library=Path(config['preload']);assert library.is_file()and hashlib.sha256(library.read_bytes()).hexdigest()==config['preload_sha256']
    shared=Path(config['shared_root']);assert shared.is_dir()and not shared.is_symlink()
    root=args.workdir.resolve();assert not root.exists();root.mkdir(parents=True,mode=0o700)
    for n,h in json.loads((frozen/'source-hashes.json').read_text()).items():
        assert hashlib.sha256((frozen/n).read_bytes()).hexdigest()==h;shutil.copyfile(frozen/n,root/n)
    for n in ('configurations.json','payloads.json'):shutil.copyfile(frozen/n,root/n)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus,MONITORED_GPUS=args.gpus,LD_PRELOAD=str(library),TEAM07_NCCL_SHM_ROOT=str(shared),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,str(root/'run_nccl_tuning222_sweep.py')],cwd=root,env=env,check=True)


if __name__=='__main__':main()
