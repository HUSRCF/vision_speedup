"""Bounded two-GPU graph/P2P feasibility test, never an inference score."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import fcntl,json,statistics,subprocess,threading,time
from pathlib import Path
import torch

physical=os.environ['MONITORED_GPUS'].split(',');assert len(physical)==4

def query():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows={}
    for line in p.stdout.splitlines():
        i,uuid,memory,util,temp,clock=[v.strip() for v in line.split(',')]
        if i in physical:rows[i]=dict(uuid=uuid,memory=int(memory),util=int(util),temperature=int(temp),clock=int(clock))
    assert len(rows)==4
    p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    contexts={r['uuid']:set() for r in rows.values()}
    for line in p.stdout.splitlines():
        uuid,pid=[v.strip() for v in line.split(',')]
        if uuid in contexts:contexts[uuid].add(int(pid))
    return rows,contexts

