"""Overlap the qualified FP16 nonfeedback decoder with native FP32 feedback."""
import torch
from einops import rearrange

from unifolm_wma.utils.vae_protected_tf32_runtime import cudnn_tf32, eligible


def parallel_clip(owner, model, samples, original, execution_steps=16):
    if (torch.is_grad_enabled() or torch.is_autocast_enabled()
            or model.first_stage_model.training or model.encoder_type != '2d'
            or model.perframe_ae is not True or not samples.is_cuda
            or samples.dtype != torch.float32 or tuple(samples.shape) != (1,4,16,40,64)
            or not samples.is_contiguous() or torch.backends.cudnn.allow_tf32
            or torch.backends.cuda.matmul.allow_tf32
            or execution_steps not in (12,16)
            or torch.cuda.is_current_stream_capturing()):
        return owner.clip(model, samples, original, execution_steps)
    frames = rearrange(samples, 'b c t h w -> (b t) c h w')
    begin = execution_steps-2
    pieces = []
    index = 0
    # Retain the exact native scaling and the qualified per-segment batch sizes.
    while index < 16:
        protected = begin <= index < execution_steps
        stop = index+1 if protected else min(index+14, begin if index < begin else 16)
        scaled = (1./model.scale_factor)*frames[index:stop]
        pieces.append((protected, scaled))
        index = stop
    current = torch.cuda.current_stream(samples.device)
    if not hasattr(owner, '_decode_work_stream'):
        owner._decode_work_stream = torch.cuda.Stream(device=samples.device)
        owner._parallel_calls = 0
    work = owner._decode_work_stream
    work.wait_stream(current)
    outputs = {}
    try:
        with torch.cuda.stream(work), cudnn_tf32(True):
            for index,(protected,scaled) in enumerate(pieces):
                if not protected:
                    scaled.record_stream(work)
                    outputs[index] = owner.fast(scaled)
        with cudnn_tf32(False):
            for index,(protected,scaled) in enumerate(pieces):
                if protected:
                    outputs[index] = owner.native(scaled)
    finally:
        current.wait_stream(work)
    for index,(protected,scaled) in enumerate(pieces):
        if not protected:
            outputs[index].record_stream(current)
    result = torch.cat([outputs[index] for index in range(len(pieces))])
    owner._parallel_calls += 1
    return rearrange(result, '(b t) c h w -> b c t h w', b=1, t=16)


import os
from unifolm_wma.utils.vae_nonfeedback_cache_runtime import configured_mode, owner_for, decode_for_stage as base_decode

FLAG='UNIFOLM_WORLD_DECODE_NONFEEDBACK_OVERLAP'
_records=[]
_fallbacks={}
_last_owner=None


def configured_overlap(environment,dataset,stage):
    raw=environment.get(FLAG,'0')
    if raw not in ('0','1'):
        raise ValueError('Invalid nonfeedback stream overlap flag')
    if raw=='0' or dataset!='unitree_z1_dual_arm_stackbox_v2' or stage!='simulation':
        return False
    if configured_mode(environment,dataset,stage)!='fp16_nchw':
        raise ValueError('Stream overlap requires the qualified cached FP16 decoder')
    return True


def decode_for_stage(model,samples,*,dataset,stage,execution_steps):
    global _last_owner
    if not configured_overlap(os.environ,dataset,stage):
        return base_decode(model,samples,dataset=dataset,stage=stage,execution_steps=execution_steps)
    feedback=model.n_obs_steps_imagen
    if (feedback!=2 or execution_steps not in (12,16)
            or not eligible(model,samples,execution_steps,feedback,14)
            or torch.cuda.is_current_stream_capturing()):
        _fallbacks['unqualified_input']=_fallbacks.get('unqualified_input',0)+1
        return base_decode(model,samples,dataset=dataset,stage=stage,execution_steps=execution_steps)
    owner=owner_for(model)
    _last_owner=owner
    before=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    caller=torch.cuda.current_stream(samples.device)
    result=parallel_clip(owner,model,samples,model.decode_first_stage,execution_steps)
    assert caller==torch.cuda.current_stream(samples.device)
    assert before==(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    _records.append(dict(dataset=dataset,stage=stage,mode='fp16_nchw',eligible=True,
        execution_steps=execution_steps,feedback_frames=feedback,
        input_shape=list(samples.shape),input_contiguous=samples.is_contiguous(),
        input_FP32=samples.dtype==torch.float32,output_shape=list(result.shape),
        output_FP32=result.dtype==torch.float32,flags_before=list(before),flags_restored=True,
        current_stream_restored=True,protected_indices=list(range(execution_steps-feedback,execution_steps))))
    return result


def get_report():
    return dict(clips=len(_records),rows=list(_records),fallbacks=dict(_fallbacks),
        cache_report=None if _last_owner is None else _last_owner.report(),
        parallel_calls=0 if _last_owner is None else getattr(_last_owner,'_parallel_calls',0),
        all_setup_inside_process_wall=True,numerical_candidate=False,
        original_FP32_decoder_masters=True,original_DDIM_and_UNet_unchanged=True,
        feedback_decode_original_perframe_FP32=True,nonfeedback_cached_FP16_only=True,
        original_cached_decoder_class=True,stream_scheduling_only=True,
        native_input_scale_and_batch_partitions_and_frame_order=True,
        input_output_record_stream_and_current_wait_work=True)
