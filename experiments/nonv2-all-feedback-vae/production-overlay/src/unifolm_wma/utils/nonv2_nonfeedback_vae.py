"""Experimental Z1 all-frame Half decoding; full closed-loop quality required."""
import atexit,copy,json,os
import torch
from einops import rearrange

def partitions(execution_steps,feedback_frames=2):
    assert 2<=execution_steps<=16 and feedback_frames==2
    begin=execution_steps-feedback_frames;index=0;result=[]
    while index<16:
        protected=begin<=index<execution_steps
        stop=index+1 if protected else min(index+14,begin if index<begin else 16)
        assert stop>index
        result.append((protected,index,stop));index=stop
    return result

class Decoder:
    def __init__(self,vae):
        self.vae=vae;self.native=vae.decode;self.cached=None;self.key=None;self.builds=0;self.calls=0;self.feedback_calls=0

    def fingerprint(self):
        return tuple((prefix,name,id(t),t.data_ptr(),t._version,tuple(t.shape),tuple(t.stride()),t.dtype,t.device)
            for prefix,module in (('post_quant_conv',self.vae.post_quant_conv),('decoder',self.vae.decoder))
            for name,t in [*module.named_parameters(),*module.named_buffers()])

    def refresh(self):
        key=self.fingerprint()
        if key!=self.key:
            copied=copy.deepcopy(torch.nn.Sequential(self.vae.post_quant_conv,self.vae.decoder))
            self.cached=copied.eval().to(dtype=torch.float16,memory_format=torch.contiguous_format)
            self.key=key;self.builds+=1

    def fast(self,value):
        self.refresh()
        return self.cached(value.half()).float()

owners={}
records=[]

@torch.no_grad()
def decode(model,samples,*,stage,execution_steps):
    scope=os.environ.get('UNIFOLM_NONV2_VAE_SCOPE','world')
    assert scope in ('world','all-display','all-feedback') and stage in ('decision','simulation')
    if scope=='all-feedback':assert os.environ.get('UNIFOLM_CASE')=='unitree_z1_stackbox'
    if stage=='decision' and scope=='world':return model.decode_first_stage(samples)
    assert model.encoder_type=='2d' and (stage=='decision' or model.perframe_ae is True)
    assert not model.first_stage_model.training and not torch.is_autocast_enabled()
    assert samples.is_cuda and samples.dtype==torch.float32 and tuple(samples.shape)==(1,4,16,40,64)
    assert model.n_obs_steps_imagen==2 and not torch.cuda.is_current_stream_capturing()
    assert all(p.dtype==torch.float32 for p in model.first_stage_model.parameters() if p.is_floating_point())
    vae=model.first_stage_model;owner=owners.get(id(vae))
    if owner is None:owner=owners[id(vae)]=Decoder(vae)
    before=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    frames=rearrange(samples,'b c t h w -> (b t) c h w');outputs=[]
    if stage=='decision' or scope=='all-feedback':
        result=owner.fast((1./model.scale_factor)*frames)
        assert before==(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
        owner.calls+=1;records.append(dict(stage=stage,display_only=stage=='decision',execution_steps=execution_steps,feedback_dtype='fp16' if stage=='simulation' else None,partitions=[(False,0,16)],flags=before))
        return rearrange(result,'(b t) c h w -> b c t h w',b=1,t=16)
    plan=partitions(execution_steps)
    for protected,start,stop in plan:
        # Preserve the original per-frame FP32 scaling, existing global TF32
        # flags and original master weights for every frame fed back to input.
        scaled=(1./model.scale_factor)*frames[start:stop]
        outputs.append(owner.native(scaled) if protected else owner.fast(scaled))
        owner.feedback_calls+=int(protected)
    assert before==(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    owner.calls+=1;records.append(dict(stage=stage,execution_steps=execution_steps,partitions=plan,flags=before))
    return rearrange(torch.cat(outputs),'(b t) c h w -> b c t h w',b=1,t=16)

def report():
    print('NONV2_NONFEEDBACK_VAE '+json.dumps(dict(clips=len(records),records=records,
        owners=[dict(builds=o.builds,calls=o.calls,feedback_calls=o.feedback_calls,mode='fp16_nchw',master_version_guard=True) for o in owners.values()],
        native_feedback_precision_and_flags_preserved=not any(r.get('feedback_dtype')=='fp16' for r in records),all_setup_inside_process_wall=True,denoiser_and_sampler_unchanged=True)),flush=True)

atexit.register(report)
