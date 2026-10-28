#!/usr/bin/env python3
"""Write canonical JSON and log timing summaries for successful and failed runs."""
import argparse
import json
import math
from pathlib import Path
import time

CASES = [('unitree_z1_stackbox',12),('unitree_z1_dual_arm_stackbox',7),
         ('unitree_z1_dual_arm_stackbox_v2',11),('unitree_z1_dual_arm_cleanup_pencils',8),
         ('unitree_g1_pack_camera',11)]


def build_result(profile_dir, exit_code, started, now=None):
    now = time.monotonic() if now is None else now
    psnr_path=profile_dir/'psnr.json'
    quality=json.loads(psnr_path.read_text()) if psnr_path.is_file() else {}
    validation_wall=quality.get('validation_wall_s',0)
    if not isinstance(validation_wall,(int,float)) or not math.isfinite(validation_wall) or validation_wall<0:
        raise ValueError('Invalid validation time')
    elapsed=round(max(0,now-started-validation_wall),6)
    execution_path=profile_dir/'execution.json'
    execution=json.loads(execution_path.read_text()) if execution_path.is_file() else {}
    rows=[]
    complete=True
    for name,rounds in CASES:
        path=profile_dir/(name+'.json')
        profile_reason=None
        try:profile=json.loads(path.read_text()) if path.is_file() else {}
        except (ValueError,OSError) as error:profile={};profile_reason=str(error)
        if not isinstance(profile,dict):profile={};profile_reason='Invalid case profile'
        timing=profile.get('total_process_wall_s')
        stages=profile.get('stages',{})
        if isinstance(timing,bool) or not isinstance(timing,(int,float)) or not math.isfinite(timing) or timing<=0:
            complete=False;timing=None
        if not isinstance(stages,dict):stages={}
        profile_complete=timing is not None and all(isinstance(stages.get(stage),dict) and stages[stage].get('count')==rounds for stage in ('decision_generation','simulation_generation'))
        case_code=execution.get(name,{}).get('exit_code',0)
        execution_ok=type(case_code) is int and case_code==0
        if not profile_complete or not execution_ok:complete=False
        q=quality.get('cases',{}).get(name,{})
        row={'case':name,'rounds':rounds,'process_wall_s':round(timing,6) if timing is not None and profile_complete and execution_ok else None,
             'observed_process_wall_s':round(timing,6) if timing is not None else None,
             'psnr_db':q.get('psnr_db'),'psnr_exact_match':q.get('exact_match') is True,
             'psnr_passed':q.get('passed') is True,'videos':q.get('video_count',0),
             'profile_complete':profile_complete,'execution_ok':execution_ok,'execution_exit_code':case_code,
             'reason':profile_reason or ('Inference exited with code '+str(case_code) if not execution_ok else 'Incomplete case profile' if not profile_complete else q.get('reason')),
             'pure_model_cuda_s':round(sum((stages.get(stage,{}) if isinstance(stages.get(stage),dict) else {}).get('cuda_s',0) for stage in ('decision_generation','simulation_generation')),6)}
        rows.append(row)
    psnr_ok=quality.get('psnr_ok') is True and all(r['psnr_passed'] for r in rows)
    accepted=exit_code==0 and complete and psnr_ok
    reason=None if accepted else quality.get('reason') or ('Incomplete case profiles' if not complete else 'PSNR validation missing or failed' if not psnr_ok else 'Benchmark exited with code '+str(exit_code))
    return {'problem':'unifolm','status':'accepted' if accepted else 'failed','exit_code':exit_code,
            'external_wall_s':elapsed,'validation_wall_s':validation_wall,'reason':reason,
            'validation':{'benchmark_exit_ok':exit_code==0,'complete_cases_ok':complete,'psnr_ok':psnr_ok},
            'review_flags':[],
            'validation_notes':['PSNR checked against shared FP32 reference videos'] if psnr_ok else [],
            'workload':{'cases':5,'rounds':[r for _,r in CASES]},'cases':rows,'psnr_validation':quality,
            'video_count':quality.get('video_count',0),'pure_model_cuda_s':sum(r['pure_model_cuda_s'] for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',required=True,type=Path)
    p.add_argument('--result',required=True,type=Path)
    p.add_argument('--exit-code',required=True,type=int)
    p.add_argument('--started',required=True,type=float)
    a=p.parse_args()
    try: result=build_result(a.profiles,a.exit_code,a.started)
    except Exception as error:
        result={'problem':'unifolm','status':'failed','exit_code':a.exit_code or 1,'reason':'Result finalization failed: '+str(error),'cases':[]}
    a.result.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.result.with_name(a.result.name+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(a.result)
    for row in result['cases']:
        if row.get('process_wall_s') is not None:
            print('TIMING case=%s process_wall_s=%.6f'%(row['case'],row['process_wall_s']),flush=True)
    if result['status']=='accepted':
        print('RUN_SUMMARY external_wall_s=%.6f validation_wall_s=%.6f'%(result['external_wall_s'],result['validation_wall_s']),flush=True)
        print('RESULT_JSON='+str(a.result),flush=True)
        print('ACCEPTED external_wall_s=%.6f'%result['external_wall_s'],flush=True)
        return 0
    print('RESULT_JSON='+str(a.result),flush=True)
    print('FAILED: '+str(result['reason']),flush=True)
    return 1


if __name__=='__main__': raise SystemExit(main())
