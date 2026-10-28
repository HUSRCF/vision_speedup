"""Run one complete accepted nonV2 case with its exact policy."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',choices=('stackbox','pencils','g1','dual'),required=True);p.add_argument('--gpu',required=True,choices=tuple('0123'));p.add_argument('--historical59',action='store_true');p.add_argument('--workdir',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parent
if a.case=='stackbox'and not a.historical59:
    command=[sys.executable,str(root.parent/'nonv2-fused-half-geglu/run_case.py'),'--gpu',a.gpu]
    if a.workdir:command+=['--workdir',str(a.workdir)]
    subprocess.run(command,check=True);sys.exit(0)
assert a.workdir is None,'--workdir is supported for current fused Z1 only'
cases={'stackbox':'unitree_z1_stackbox','pencils':'unitree_z1_dual_arm_cleanup_pencils','g1':'unitree_g1_pack_camera','dual':'unitree_z1_dual_arm_stackbox'}
flags=root/(cases[a.case]+'-flags.json');assert flags.is_file(),'This case has not been adopted in this snapshot'
env={k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_')};env.update(json.loads(flags.read_text()))
subprocess.run(['bash','scripts/team07_run.sh','case-'+a.case,a.gpu],cwd=root/'unifolm',env=env,check=True)
