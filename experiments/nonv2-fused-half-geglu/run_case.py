"""Prepare an isolated exact60-source checkout and run the complete Z1 case."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',choices=tuple('0123'),required=True)
    parser.add_argument('--workdir',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();here=Path(__file__).resolve().parent
    work=(args.workdir or Path.home()/'competition'/('unifolm-fused-half-review-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    assert not work.exists(),'Choose a fresh workdir'
    source=work/'source'
    subprocess.run([sys.executable,str(here/'materialize.py'),'--output',str(source)],check=True)
    # Keep the actual tested whole-attempt guard and output/provenance policy.
    for name in('scripts/team07_run.sh','scripts/wait_clean_run.sh'):
        shutil.copyfile(here/'full-candidate-r1'/name,source/name)
    # The original wrapper records git provenance. An independent repository
    # avoids treating the review repository's unrelated files as deletions.
    subprocess.run(['git','init','--quiet',str(source)],check=True)
    subprocess.run(['git','-C',str(source),'add','-f','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=UnifoLM reproduction','-c','user.email=reproduction@localhost','commit','--quiet','-m','Freeze qualified60-source Z1 reproduction'],check=True)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True,source_count=60,workdir=str(work),source_directory=str(source),no_GPU_execution=True)))
        return
    env={k:v for k,v in os.environ.items()if not k.startswith('UNIFOLM_')and k!='TRITON_CACHE_MANAGER'}
    env.update(json.loads((here/'flags.json').read_text()))
    env.update(EXPERIMENT_RUN_ROOT=str(work/'runs'),EXPERIMENT_OUTPUT_ROOT=str(work/'outputs'),WAIT_RUN_MODE='case-stackbox',WAIT_GPU_CANDIDATES=args.gpu,WAIT_TIMEOUT_SECONDS='14400',RESERVATION_HOLD_SECONDS='120',IDLE_GRACE_SECONDS='0',CLEAN_POLL_SECONDS='2',LATE_CONTAMINATION_SECONDS='0',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['bash',str(source/'scripts/wait_clean_run.sh')],cwd=source,env=env,check=True)


if __name__=='__main__':main()
