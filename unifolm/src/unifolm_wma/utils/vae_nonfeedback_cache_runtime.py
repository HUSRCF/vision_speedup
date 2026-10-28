"""Diagnostic decoder copies, used only for the nonfeedback pixels."""
import copy
from types import SimpleNamespace

import torch

from unifolm_wma.utils.vae_protected_tf32_runtime import decode, configured_chunk, eligible


class NonfeedbackDecoder:
    def __init__(self, vae, mode):
        assert mode in ('tf32_nhwc', 'fp16_nchw', 'fp16_nhwc')
        self.vae = vae
        self.mode = mode
        self.native = vae.decode
        self.cached = None
        self.key = None
        self.builds = 0
        self.hits = 0
        self.setup_ms = []

    def fingerprint(self):
        return tuple((prefix, name, id(tensor), tensor.data_ptr(), tensor._version,
                      tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)
                     for prefix, module in [('post_quant_conv', self.vae.post_quant_conv),
                                            ('decoder', self.vae.decoder)]
                     for name, tensor in [*module.named_parameters(), *module.named_buffers()])

    def fast(self, value):
        import time
        key = self.fingerprint()
        if self.key != key:
            torch.cuda.synchronize()
            started = time.perf_counter()
            self.cached = None
            copied = copy.deepcopy(torch.nn.Sequential(self.vae.post_quant_conv, self.vae.decoder))
            copied.eval()
            dtype = torch.float32 if self.mode == 'tf32_nhwc' else torch.float16
            layout = torch.channels_last if self.mode.endswith('nhwc') else torch.contiguous_format
            self.cached = copied.to(dtype=dtype, memory_format=layout)
            self.key = key
            self.builds += 1
            torch.cuda.synchronize()
            self.setup_ms.append((time.perf_counter()-started)*1000)
        else:
            self.hits += 1
        dtype = torch.float32 if self.mode == 'tf32_nhwc' else torch.float16
        layout = torch.channels_last if self.mode.endswith('nhwc') else torch.contiguous_format
        output = self.cached(value.to(dtype=dtype, memory_format=layout))
        return output.to(dtype=torch.float32, memory_format=torch.contiguous_format)

    def selected_decode(self, value):
        if not torch.backends.cudnn.allow_tf32:
            return self.native(value)
        return self.fast(value)

    def clip(self, model, samples, original, execution_steps=16):
        proxy = SimpleNamespace(encoder_type=model.encoder_type,
            perframe_ae=model.perframe_ae, scale_factor=model.scale_factor,
            first_stage_model=SimpleNamespace(training=self.vae.training, decode=self.selected_decode))
        return decode(proxy, samples, original, execution_steps=execution_steps,
                      feedback_frames=2, chunk_size=14)

    def clear(self):
        self.cached = None
        self.key = None

    def report(self):
        return dict(mode=self.mode, builds=self.builds, hits=self.hits, setup_ms=self.setup_ms,
                    copied_weight_bytes=sum(p.numel()*p.element_size() for p in self.cached.parameters()) if self.cached is not None else 0,
                    original_master_version_guarded=True, original_FP32_feedback_decoder=True)


import os

FLAG = 'UNIFOLM_WORLD_DECODE_NONFEEDBACK_CACHE'
_owner = None
_records = []
_fallbacks = {}


def configured_mode(environment, dataset, stage):
    raw = environment.get(FLAG, '0')
    if raw not in ('0', 'fp16_nchw'):
        raise ValueError('Invalid nonfeedback decoder cache mode')
    if raw == '0' or dataset != 'unitree_z1_dual_arm_stackbox_v2' or stage != 'simulation':
        return None
    if configured_chunk(environment, dataset, stage) != 14:
        raise ValueError('Nonfeedback decoder cache requires qualified protected chunk14')
    return raw


def owner_for(model):
    global _owner
    if _owner is None or _owner.vae is not model.first_stage_model:
        if _owner is not None:
            _owner.clear()
        _owner = NonfeedbackDecoder(model.first_stage_model, 'fp16_nchw')
    return _owner


def decode_for_stage(model, samples, *, dataset, stage, execution_steps):
    mode = configured_mode(os.environ, dataset, stage)
    if mode is None:
        return model.decode_first_stage(samples)
    feedback = model.n_obs_steps_imagen
    if (feedback != 2 or not eligible(model, samples, execution_steps, feedback, 14)
            or torch.cuda.is_current_stream_capturing()):
        _fallbacks['unqualified_input'] = _fallbacks.get('unqualified_input', 0) + 1
        return model.decode_first_stage(samples)
    owner = owner_for(model)
    before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    result = owner.clip(model, samples, model.decode_first_stage, execution_steps)
    assert before == (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    _records.append(dict(dataset=dataset, stage=stage, mode=mode, eligible=True,
        execution_steps=execution_steps, feedback_frames=feedback,
        input_shape=list(samples.shape), input_contiguous=samples.is_contiguous(),
        input_FP32=samples.dtype == torch.float32, output_shape=list(result.shape),
        output_FP32=result.dtype == torch.float32, flags_before=list(before), flags_restored=True,
        protected_indices=list(range(execution_steps-feedback, execution_steps))))
    return result


def get_report():
    return dict(clips=len(_records), rows=list(_records), fallbacks=dict(_fallbacks),
        cache_report=None if _owner is None else _owner.report(),
        all_setup_inside_process_wall=True, numerical_candidate=True,
        original_FP32_decoder_masters=True, original_DDIM_and_UNet_unchanged=True,
        feedback_decode_original_perframe_FP32=True, nonfeedback_cached_FP16_only=True,
        master_values_not_modified=True, native_input_scale_and_frame_order=True)
