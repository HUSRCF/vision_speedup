"""Run one complete accepted nonV2 case with its exact policy."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',choices=('stackbox','pencils','g1','dual'),required=True);p.add_argument('--gpu',required=True,choices=tuple('0123'));a=p.parse_args()
root=Path(__file__).resolve().parent
cases={'stackbox':'unitree_z1_stackbox','pencils':'unitree_z1_dual_arm_cleanup_pencils','g1':'unitree_g1_pack_camera','dual':'unitree_z1_dual_arm_stackbox'}
flags=root/(cases[a.case]+'-flags.json');assert flags.is_file(),'This case has not been adopted in this snapshot'
env={k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_')};env.update(json.loads(flags.read_text()))
subprocess.run(['bash','scripts/team07_run.sh','case-'+a.case,a.gpu],cwd=root/'unifolm',env=env,check=True)
