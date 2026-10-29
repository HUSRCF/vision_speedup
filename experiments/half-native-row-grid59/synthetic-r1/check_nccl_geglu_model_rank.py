import os,json,runpy,sys,time
from pathlib import Path
rank=int(os.environ['LOCAL_RANK'])
os.environ['CUDA_VISIBLE_DEVICES']=os.environ['MONITORED_GPUS'].split(',')[rank]
import torch
import torch.distributed as dist
from check_dual_gpu_graph import query,physical
torch.cuda.set_device(0)
hold=torch.empty(64*2**20,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
rows,contexts=query();root=set(json.loads(os.environ['ROOT_NVML_OWNERS']))
own=contexts[rows[physical[rank]]['uuid']]-root;assert len(own)==1, 'rank CUDA context ownership unresolved'
workspace=Path.cwd();(workspace/('rank'+str(rank)+'-ready.json')).write_text(json.dumps(dict(rank=rank,host_pids=sorted(own))))
until=time.monotonic()+60
while not (workspace/'go').exists():assert time.monotonic()<until;time.sleep(.02)
dist.init_process_group('nccl');assert dist.get_world_size()==2
script=Path(os.environ['EXPERIMENT_ROOT'])/'scripts/audit_half_native_row_grid59.py'
sys.argv=[str(script)];runpy.run_path(str(script),run_name='__main__')
report=json.loads((workspace/('rank'+str(rank)+'-report.json')).read_text())
assert report['synthetic_only']and report['all_supported_cases_bitwise']and len(report['rows'])==10
assert report['no_kernel_argument_or_address_exports']
torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
print('NCCL_GEGLU_MODEL_RANK_CHECK passed=true rank='+str(rank),flush=True)
