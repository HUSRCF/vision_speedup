import json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from unifolm_wma.models.ddpms import LatentDiffusion
from unifolm_wma.utils.utils import instantiate_from_config
from unifolm_wma.utils.vae_nonfeedback_overlap_runtime import FLAG,decode_for_stage,get_report
from unifolm_wma.utils.vae_nonfeedback_cache_runtime import owner_for,decode_for_stage as sequential_decode,NonfeedbackDecoder

def exact(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))

@torch.no_grad()
def main():
 torch.manual_seed(1569);torch.use_deterministic_algorithms(True)
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
 os.environ.update({**{'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_FULL_DENOISER_FP16': '0', 'UNIFOLM_BATCH_DECISION_DECODE': '1', 'UNIFOLM_STAGE_TF32': 'decision-cudnn', 'UNIFOLM_STAGE_TF32_ALL_CASES': '0', 'UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS': '0'},'UNIFOLM_WORLD_DECODE_PROTECTED_TF32_CHUNK':'14','UNIFOLM_WORLD_DECODE_NONFEEDBACK_CACHE':'fp16_nchw',FLAG:'1'})
 def trial(value,steps=16):
  return decode_for_stage(model,value,dataset='unitree_z1_dual_arm_stackbox_v2',stage='simulation',execution_steps=steps)
 def baseline(value,steps=16):
  return sequential_decode(model,value,dataset='unitree_z1_dual_arm_stackbox_v2',stage='simulation',execution_steps=steps)
 video=torch.randn(1,4,16,40,64,device='cuda')*.2;saved_input=video.clone()
 masters=[p for n,p in vae.named_parameters() if n.startswith(('post_quant_conv.','decoder.'))];saved=[p.clone() for p in masters]
 keys=list(vae.state_dict());ptrs=[p.data_ptr() for p in vae.parameters()];rng=torch.cuda.get_rng_state();checks=[]
 initial=model.decode_first_stage(video)
 for label,steps in [('initial',16),('short_execution',12),('changed_input',16),('restored_input',16),('changed_all_masters',16),('restored_all_masters',16),('zero_all_masters',16),('zero_then_restore',16)]:
  if label=='changed_input':video.add_(.00015)
  elif label=='restored_input':video.copy_(saved_input)
  elif label in ['changed_all_masters','restored_all_masters','zero_all_masters','zero_then_restore']:
   for p,saved_p in zip(masters,saved):
    if label=='changed_all_masters':p.add_(.00013)
    elif label=='zero_all_masters':p.zero_()
    else:p.copy_(saved_p)
  owner=owner_for(model);before=owner.builds
  expected=baseline(video,steps);actual=trial(video,steps);exact(actual,expected)
  exact(actual[:,:,steps-2:steps],model.decode_first_stage(video)[:,:,steps-2:steps])
  retained=actual.clone();other=trial(video+.00017,steps);exact(actual,retained)
  assert actual.data_ptr()!=other.data_ptr()
  if 'masters' in label or label=='zero_then_restore':assert owner.builds==before+1
  checks.append(dict(case=label,execution_steps=steps,all16_pixels_int32_exact=True,feedback_native_int32_exact=True,independent_outputs=True,master_cache_refreshed=True))
 exact(model.decode_first_stage(video),initial)
 for p,saved_p in zip(masters,saved):exact(p,saved_p)
 owner=owner_for(model);assert owner.__class__ is NonfeedbackDecoder
 default=torch.cuda.current_stream();caller=torch.cuda.Stream();caller.wait_stream(default)
 stream_checks=[]
 for steps in [16,12]:
  with torch.cuda.stream(caller):
   video.record_stream(caller);before=torch.cuda.current_stream().cuda_stream
   actual=trial(video,steps)
   assert torch.cuda.current_stream().cuda_stream==before
   assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
  default.wait_stream(caller);actual.record_stream(default);exact(actual,baseline(video,steps))
  stream_checks.append(dict(execution_steps=steps,nondefault_caller_restored=True,all16_pixels_int32_exact=True))
 exception_checks=[]
 for name in ['fast','native']:
  old=getattr(owner,name)
  def raising(value):
   torch.cuda._sleep(2000000)
   raise RuntimeError('intentional '+name+' exception')
  setattr(owner,name,raising)
  with torch.cuda.stream(caller):
   before=torch.cuda.current_stream().cuda_stream
   try:trial(video)
   except RuntimeError as error:assert str(error)=='intentional '+name+' exception'
   else:raise AssertionError('exception missing')
   finally:setattr(owner,name,old)
   assert torch.cuda.current_stream().cuda_stream==before
   assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
   recovered=trial(video)
  default.wait_stream(caller);recovered.record_stream(default);exact(recovered,baseline(video))
  exception_checks.append(dict(path=name,nondefault_caller_restored=True,flags_restored=True,recovery_all16_pixels_int32_exact=True))
 stress=[]
 for index in range(6):
  steps=16 if index%2==0 else 12
  with torch.cuda.stream(caller):
   local=video.clone().add_(index*.00003)
   expected=baseline(local,steps);actual=trial(local,steps)
   del local
   churn=[torch.empty((1,4,16,40,64),device='cuda').fill_(index+9.) for _ in range(8)]
  # Reuse the fast stream after its outputs' Python references have been freed.
  # record_stream(current) must retain those allocations until current's cat reads them.
  with torch.cuda.stream(owner._decode_work_stream):
   garbage=[torch.empty((14,3,320,512),device='cuda').fill_(index+13.) for _ in range(2)]
  default.wait_stream(caller);default.wait_stream(owner._decode_work_stream)
  actual.record_stream(default);expected.record_stream(default);exact(actual,expected)
  assert torch.cuda.current_stream()==default
  stress.append(dict(execution_steps=steps,freed_input_and_work_output_allocator_churn=True,all16_pixels_int32_exact=True))
  del churn,garbage,actual,expected
 native=owner.native;fast=owner.fast;trace=[]
 def traced_native(value):
  trace.append(dict(path='native_feedback',batch=value.shape[0],current_stream=torch.cuda.current_stream().cuda_stream,input_dtype=str(value.dtype),cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32))
  return native(value)
 def traced_fast(value):
  output=fast(value)
  trace.append(dict(path='cached_nonfeedback',batch=value.shape[0],current_stream=torch.cuda.current_stream().cuda_stream,weight_dtype=str(next(owner.cached.parameters()).dtype),output_dtype=str(output.dtype),cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32))
  return output
 owner.native=traced_native;owner.fast=traced_fast
 try:actual=trial(video)
 finally:owner.native=native;owner.fast=fast
 assert [r['batch'] for r in trace if r['path']=='native_feedback']==[1,1]
 assert [r['batch'] for r in trace if r['path']=='cached_nonfeedback']==[14]
 assert all(r['current_stream']==default.cuda_stream and not r['cudnn_tf32'] and r['input_dtype']=='torch.float32' for r in trace if r['path']=='native_feedback')
 assert all(r['current_stream']==owner._decode_work_stream.cuda_stream and r['weight_dtype']=='torch.float16' and r['output_dtype']=='torch.float32' for r in trace if r['path']=='cached_nonfeedback')
 assert all(not r['matmul_tf32'] for r in trace)
 assert torch.equal(rng,torch.cuda.get_rng_state()) and keys==list(vae.state_dict()) and ptrs==[p.data_ptr() for p in vae.parameters()]
 assert torch.cuda.current_stream()==default
 report=get_report();assert report['fallbacks']=={} and report['cache_report']['builds']==5
 assert report['parallel_calls']==report['clips']==27
 output=dict(passed=True,checks=checks,stream_checks=stream_checks,exception_checks=exception_checks,allocator_stress=stress,trace=trace,report=report,masters=140,original_baseline_restored_int32_exact=True,RNG_unchanged=True,master_pointers_values_keys_unchanged=True,qualified_cached_decoder_class_retained=True,nondefault_stream_explicit_success_and_both_exceptions=True,allocator_lifetime_stress_all16_pixels_exact=True)
 Path('vae-nonfeedback-overlap-production-gpu.json').write_text(json.dumps(output,indent=2)+'\n')
 print('VAE_NONFEEDBACK_OVERLAP_PRODUCTION_CHECK passed=true fixtures=8',flush=True)

if __name__=='__main__':main()
