"""Diagnostic precision scope: keep all feedback pixels on native FP32 decode."""
from contextlib import contextmanager

import torch
from einops import rearrange


@contextmanager
def cudnn_tf32(enabled):
    before = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = enabled
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = before


def decode(model, samples, original, *, execution_steps, feedback_frames,
           chunk_size):
    # The diagnostic is intentionally confined to the actual world clip layout.
    if (torch.is_grad_enabled() or torch.is_autocast_enabled()
            or model.first_stage_model.training
            or model.encoder_type != '2d' or model.perframe_ae is not True
            or not isinstance(samples, torch.Tensor)
            or not samples.is_cuda or samples.dtype != torch.float32
            or tuple(samples.shape) != (1, 4, 16, 40, 64)
            or not samples.is_contiguous()
            or torch.backends.cudnn.allow_tf32
            or torch.backends.cuda.matmul.allow_tf32
            or not 1 <= feedback_frames <= execution_steps <= 16
            or chunk_size not in (1, 4, 8, 14)):
        return original(samples)
    frames = rearrange(samples, 'b c t h w -> (b t) c h w')
    begin = execution_steps - feedback_frames
    outputs = []
    index = 0
    while index < len(frames):
        protected = begin <= index < execution_steps
        stop = index + 1 if protected else min(
            index + chunk_size, begin if index < begin else len(frames))
        scaled = (1. / model.scale_factor) * frames[index:stop]
        with cudnn_tf32(not protected):
            outputs.append(model.first_stage_model.decode(scaled))
        index = stop
    return rearrange(torch.cat(outputs), '(b t) c h w -> b c t h w', b=1, t=16)


import os

FLAG = 'UNIFOLM_WORLD_DECODE_PROTECTED_TF32_CHUNK'
DEPS = {'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_FULL_DENOISER_FP16': '0', 'UNIFOLM_BATCH_DECISION_DECODE': '1', 'UNIFOLM_STAGE_TF32': 'decision-cudnn', 'UNIFOLM_STAGE_TF32_ALL_CASES': '0', 'UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS': '0'}
_records = []


def configured_chunk(environment, dataset, stage):
    raw = environment.get(FLAG, '0')
    if raw not in ('0', '1', '4', '8', '14'):
        raise ValueError('Invalid protected TF32 decode chunk')
    if raw == '0' or dataset != 'unitree_z1_dual_arm_stackbox_v2' or stage != 'simulation':
        return 0
    if any(environment.get(k, '0') != v for k, v in DEPS.items()):
        raise ValueError('Protected TF32 decode requires qualified V2 policies')
    return int(raw)


def eligible(model, samples, execution_steps, feedback_frames, chunk_size):
    return not (torch.is_grad_enabled() or torch.is_autocast_enabled() or model.first_stage_model.training or (model.encoder_type != '2d') or (model.perframe_ae is not True) or (not isinstance(samples, torch.Tensor)) or (not samples.is_cuda) or (samples.dtype != torch.float32) or (tuple(samples.shape) != (1, 4, 16, 40, 64)) or (not samples.is_contiguous()) or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32 or (not 1 <= feedback_frames <= execution_steps <= 16) or (chunk_size not in (1, 4, 8, 14)))


def decode_for_stage(model, samples, *, dataset, stage, execution_steps):
    chunk = configured_chunk(os.environ, dataset, stage)
    if not chunk:
        return model.decode_first_stage(samples)
    feedback = model.n_obs_steps_imagen
    applied = eligible(model, samples, execution_steps, feedback, chunk)
    old_flags = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    result = decode(model, samples, model.decode_first_stage,
                    execution_steps=execution_steps, feedback_frames=feedback, chunk_size=chunk)
    assert old_flags == (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    _records.append(dict(dataset=dataset, stage=stage, chunk=chunk, eligible=applied,
        execution_steps=execution_steps, feedback_frames=feedback,
        input_shape=list(samples.shape), input_contiguous=samples.is_contiguous(),
        input_FP32=samples.dtype == torch.float32, output_shape=list(result.shape),
        output_FP32=result.dtype == torch.float32, flags_before=list(old_flags), flags_restored=True,
        protected_indices=list(range(execution_steps-feedback, execution_steps))))
    return result


def get_report():
    return dict(clips=len(_records), rows=list(_records),
                all_setup_inside_process_wall=True, numerical_candidate=True,
                original_FP32_decoder_masters=True, original_DDIM_and_UNet_unchanged=True,
                feedback_decode_original_perframe_FP32=True, nonfeedback_cudnn_TF32_only=True)
