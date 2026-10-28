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
flags=json.loads((workspace/'flags.json').read_text());os.environ.update(flags)
base=Path(os.environ['CANONICAL_ROOT']);os.chdir(base)
directory=workspace/('rank'+str(rank)+'-decision')
script=workspace/'scripts/profile_nccl_geglu_209_pair.py'
sys.argv=[str(script),'--diagnostic-dir',str(directory),'--profile-stage','decision',
 '--seed','123','--ckpt_path','/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt','--config','configs/inference/world_model_interaction.yaml',
 '--dataset','unitree_z1_dual_arm_stackbox_v2','--savedir',str(directory)+'-unused','--bs','1','--height','320','--width','512',
 '--unconditional_guidance_scale','1.0','--ddim_steps','50','--ddim_eta','1.0','--video_length','16','--frame_stride','4','--exe_steps','16',
 '--n_iter','1','--timestep_spacing','uniform_trailing','--guidance_rescale','.7','--perframe_ae',
 '--input_cache_dir','/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs','--require_input_cache']
runpy.run_path(str(script),run_name='__main__')
report=json.loads((directory/'comparison.json').read_text())
assert len(report['cases'])==6 and all(e['signedzero_bits'] for c in report['cases'] for e in c['errors'])
(workspace/('rank'+str(rank)+'-report.json')).write_text(json.dumps(report,indent=2)+'\n')
torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
print('NCCL_GEGLU_MODEL_RANK_CHECK passed=true rank='+str(rank),flush=True)
