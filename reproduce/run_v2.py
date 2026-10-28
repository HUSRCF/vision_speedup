"""Run the original full-workload V2 launcher with recorded production flags."""
import argparse,json,os,subprocess
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();assert args.gpu>=0
    root=Path(__file__).resolve().parents[1]
    flags=json.loads((root/'reproduce/v2-effective-flags.json').read_text())
    assert flags['UNIFOLM_WORLD_HEAD_FILTER_CACHE']=='1'
    env={k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_')}
    env.update(flags)
    command=['bash',str(root/'unifolm/scripts/team07_run.sh'),'case-v2',str(args.gpu)]
    raise SystemExit(subprocess.call(command,cwd=root/'unifolm',env=env))

if __name__=='__main__':main()
