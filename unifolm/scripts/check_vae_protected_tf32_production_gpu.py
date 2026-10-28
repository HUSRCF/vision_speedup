import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from unifolm_wma.models.ddpms import LatentDiffusion
from unifolm_wma.utils.utils import instantiate_from_config
from unifolm_wma.utils.vae_protected_tf32_runtime import FLAG,DEPS,decode_for_stage,get_report

def exact(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))

@torch.no_grad()
def main():
 torch.manual_seed(1565);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 cfg=OmegaConf.load('configs/inference/world_model_interaction.yaml').model.params
 vae=instantiate_from_config(cfg.first_stage_config).eval()
 raw=torch.load('/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',map_location='cpu',mmap=True)
 state=raw.get('state_dict',raw);prefix='first_stage_model.'
 vae.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
 del raw,state;vae.cuda()
 model=SimpleNamespace(encoder_type='2d',perframe_ae=True,scale_factor=cfg.scale_factor,first_stage_model=vae,n_obs_steps_imagen=2)
 model.decode_first_stage=lambda x:LatentDiffusion.decode_core(model,x)
 os.environ.update({**DEPS,FLAG:'14'})
 video=torch.randn(1,4,16,40,64,device='cuda')*.2;original=video.clone()
 masters=[p for n,p in vae.named_parameters() if n.startswith(('post_quant_conv.','decoder.'))];saved=[p.clone() for p in masters]
 keys=list(vae.state_dict());ptrs=[p.data_ptr() for p in vae.parameters()];rng=torch.cuda.get_rng_state();checks=[]
 initial=model.decode_first_stage(video)
 for label,steps in [('initial',16),('short_execution',12),('changed_input',16),('restored_input',16),('changed_all_masters',16),('restored_all_masters',16),('zero_all_masters',16),('zero_then_restore',16)]:
  if label=='changed_input':video.add_(.00015)
  elif label=='restored_input':video.copy_(original)
  elif label in ['changed_all_masters','restored_all_masters','zero_all_masters','zero_then_restore']:
   for p,saved_p in zip(masters,saved):
    if label=='changed_all_masters':p.add_(.00013)
    elif label=='zero_all_masters':p.zero_()
    else:p.copy_(saved_p)
  expected=model.decode_first_stage(video)
  actual=decode_for_stage(model,video,dataset='unitree_z1_dual_arm_stackbox_v2',stage='simulation',execution_steps=steps)
  exact(actual[:,:,steps-2:steps],expected[:,:,steps-2:steps])
  retained=actual.clone();other=decode_for_stage(model,video+.00017,dataset='unitree_z1_dual_arm_stackbox_v2',stage='simulation',execution_steps=steps)
  exact(actual,retained);assert other.data_ptr()!=actual.data_ptr()
  assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
  checks.append(dict(case=label,execution_steps=steps,feedback_int32_exact=True,independent_outputs=True))
 exact(model.decode_first_stage(video),initial)
 for p,saved_p in zip(masters,saved):exact(p,saved_p)
 # Trace an actual helper call to prove per-frame math modes and batch partition.
 native=vae.decode;trace=[]
 def traced(value):
  trace.append(dict(batch=value.shape[0],cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32))
  return native(value)
 vae.decode=traced
 try:result=decode_for_stage(model,video,dataset='unitree_z1_dual_arm_stackbox_v2',stage='simulation',execution_steps=16)
 finally:vae.decode=native
 assert sum(r['batch'] for r in trace if r['cudnn_tf32'])==14
 assert [r['batch'] for r in trace if not r['cudnn_tf32']]==[1,1]
 assert all(not r['matmul_tf32'] for r in trace)
 assert torch.equal(rng,torch.cuda.get_rng_state()) and keys==list(vae.state_dict()) and ptrs==[p.data_ptr() for p in vae.parameters()]
 report=get_report();assert report['clips']==17 and all(r['eligible'] for r in report['rows'])
 output=dict(passed=True,checks=checks,report=report,trace=trace,masters=140,original_baseline_restored_int32_exact=True,RNG_unchanged=True,master_pointers_values_keys_unchanged=True)
 Path('vae-protected-tf32-production-gpu.json').write_text(json.dumps(output,indent=2)+'\n')
 print('VAE_PROTECTED_TF32_PRODUCTION_CHECK passed=true fixtures=8',flush=True)

if __name__=='__main__':main()
