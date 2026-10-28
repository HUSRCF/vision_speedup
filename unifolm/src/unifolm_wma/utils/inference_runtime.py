"""Explicit inference math modes; every mode still needs full video validation."""
from contextlib import contextmanager
import os
import torch


TF32_CANDIDATE_CASES = {
    "unitree_z1_stackbox",
    "unitree_z1_dual_arm_stackbox",
    "unitree_z1_dual_arm_cleanup_pencils",
    "unitree_g1_pack_camera",
}
STAGE_TF32_CASE = "unitree_z1_dual_arm_stackbox_v2"
VALIDATED_FP16_CASES = {
    "unitree_z1_stackbox",
    "unitree_z1_dual_arm_stackbox",
    "unitree_z1_dual_arm_cleanup_pencils",
}
FUSED_QKV_CANDIDATE_CASES = {
    "unitree_z1_stackbox",
    "unitree_z1_dual_arm_cleanup_pencils",
}
CROSS_ATTENTION_CANDIDATE_CASES = FUSED_QKV_CANDIDATE_CASES

STAGE_TF32_MODES = {
    "none",
    "decision-cudnn",
    "simulation-cudnn",
    "decision-matmul",
    "simulation-matmul",
}

STAGE_CUDNN_BENCHMARK_MODES = {"none", "decision", "simulation"}
_STAGE_CUDNN_BENCHMARK_DONE = set()


def full_denoiser_fp16_enabled(mode: str, dataset: str) -> bool:
    if mode not in {"0", "1", "non-v2", "validated"}:
        raise ValueError(
            "UNIFOLM_FULL_DENOISER_FP16 must be 0, 1, non-v2, or validated")
    return (
        mode == "1" or
        (mode == "non-v2" and dataset != STAGE_TF32_CASE) or
        (mode == "validated" and dataset in VALIDATED_FP16_CASES)
    )


def _stage_tf32_mode():
    mode = os.environ.get("UNIFOLM_STAGE_TF32", "none")
    if mode not in STAGE_TF32_MODES:
        raise ValueError(
            f"Unknown UNIFOLM_STAGE_TF32={mode!r}; "
            f"expected {sorted(STAGE_TF32_MODES)}")
    return mode


def stage_tf32_ddim_prefix_steps() -> int:
    """Number of leading DDIM iterations that use the selected stage TF32 mode."""
    value = os.environ.get("UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS", "0")
    try:
        steps = int(value)
    except ValueError as error:
        raise ValueError(
            "UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS must be a non-negative integer"
        ) from error
    if steps < 0:
        raise ValueError(
            "UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS must be a non-negative integer")
    return steps


def _stage_tf32_applies(dataset: str) -> bool:
    all_cases = os.environ.get("UNIFOLM_STAGE_TF32_ALL_CASES", "0")
    if all_cases not in {"0", "1"}:
        raise ValueError("UNIFOLM_STAGE_TF32_ALL_CASES must be 0 or 1")
    return dataset == STAGE_TF32_CASE or all_cases == "1"


def _stage_cudnn_benchmark_mode():
    mode = os.environ.get("UNIFOLM_STAGE_CUDNN_BENCHMARK", "none")
    if mode not in STAGE_CUDNN_BENCHMARK_MODES:
        raise ValueError(
            f"Unknown UNIFOLM_STAGE_CUDNN_BENCHMARK={mode!r}; "
            f"expected {sorted(STAGE_CUDNN_BENCHMARK_MODES)}")
    return mode


def _stage_cudnn_benchmark_applies(dataset: str) -> bool:
    all_cases = os.environ.get("UNIFOLM_STAGE_CUDNN_BENCHMARK_ALL_CASES", "0")
    if all_cases not in {"0", "1"}:
        raise ValueError(
            "UNIFOLM_STAGE_CUDNN_BENCHMARK_ALL_CASES must be 0 or 1")
    return dataset == STAGE_TF32_CASE or all_cases == "1"


@contextmanager
def skip_parameter_initialization(enabled: bool):
    """Skip random fills for parameters immediately overwritten by a strict checkpoint.

    The model is still built with ordinary CPU tensors, so constructor shape
    probes and device-sensitive setup keep their normal behavior. The caller
    must load a complete checkpoint with ``strict=True`` before inference.
    """
    if not enabled:
        yield
        return
    names = (
        "uniform_", "normal_", "constant_", "ones_", "zeros_",
        "xavier_uniform_", "xavier_normal_", "kaiming_uniform_",
        "kaiming_normal_", "trunc_normal_", "orthogonal_",
    )
    originals = {name: getattr(torch.nn.init, name) for name in names}

    def unchanged(tensor, *args, **kwargs):
        return tensor

    try:
        for name in names:
            setattr(torch.nn.init, name, unchanged)
        yield
    finally:
        for name, original in originals.items():
            setattr(torch.nn.init, name, original)


def configure_math_mode(mode: str, dataset: str) -> None:
    modes = {"fp32", "tf32", "cudnn-tf32", "matmul-tf32", "selective-tf32"}
    if mode not in modes:
        raise ValueError(f"Unknown UNIFOLM_MATH_MODE={mode!r}; expected {sorted(modes)}")
    effective = mode
    if mode == "selective-tf32":
        # QUICKSTART documents failures on stackbox_v2 and an earlier passing
        # run for the four candidates. Keep v2 and unknown datasets in FP32.
        effective = "tf32" if dataset in TF32_CANDIDATE_CASES else "fp32"
    torch.backends.cudnn.allow_tf32 = effective in {"tf32", "cudnn-tf32"}
    torch.backends.cuda.matmul.allow_tf32 = effective in {"tf32", "matmul-tf32"}
    requested_stage_mode = _stage_tf32_mode()
    stage_mode = (requested_stage_mode
                  if _stage_tf32_applies(dataset) else "none")
    ddim_prefix_steps = stage_tf32_ddim_prefix_steps()
    if ddim_prefix_steps and requested_stage_mode == "none":
        raise ValueError(
            "UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS requires UNIFOLM_STAGE_TF32")
    if stage_mode != "none" and effective != "fp32":
        raise ValueError("UNIFOLM_STAGE_TF32 requires global FP32 math mode")
    benchmark = os.environ.get("UNIFOLM_CUDNN_BENCHMARK", "0")
    if benchmark not in {"0", "1"}:
        raise ValueError("UNIFOLM_CUDNN_BENCHMARK must be 0 or 1")
    torch.backends.cudnn.benchmark = benchmark == "1"
    benchmark_limit = os.environ.get("UNIFOLM_CUDNN_BENCHMARK_LIMIT", "10")
    try:
        benchmark_limit = int(benchmark_limit)
    except ValueError as error:
        raise ValueError(
            "UNIFOLM_CUDNN_BENCHMARK_LIMIT must be a non-negative integer") from error
    if benchmark_limit < 0:
        raise ValueError(
            "UNIFOLM_CUDNN_BENCHMARK_LIMIT must be a non-negative integer")
    torch.backends.cudnn.benchmark_limit = benchmark_limit
    print(f"INFERENCE_MATH requested={mode} effective={effective} case={dataset} "
          f"cudnn_tf32={torch.backends.cudnn.allow_tf32} "
          f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn_benchmark={torch.backends.cudnn.benchmark} "
          f"cudnn_benchmark_limit={torch.backends.cudnn.benchmark_limit}",
          flush=True)
    print(f"STAGE_TF32 requested={requested_stage_mode} "
          f"effective={stage_mode} case={dataset} "
          f"ddim_prefix_steps={ddim_prefix_steps}", flush=True)
    requested_benchmark_stage = _stage_cudnn_benchmark_mode()
    benchmark_stage = (requested_benchmark_stage
                       if _stage_cudnn_benchmark_applies(dataset) else "none")
    if benchmark_stage != "none" and torch.backends.cudnn.benchmark:
        raise ValueError(
            "UNIFOLM_STAGE_CUDNN_BENCHMARK requires global cuDNN benchmark off")
    print(f"STAGE_CUDNN_BENCHMARK requested={requested_benchmark_stage} "
          f"effective={benchmark_stage} case={dataset}", flush=True)


@contextmanager
def generation_math(stage: str, dataset: str):
    """Optionally capture one generation stage with a narrower TF32 policy.

    Decision-making and world simulation have distinct CUDA Graph signatures.
    This permits one graph to use faster cuDNN convolutions while the other
    retains FP32.  Backend flags are restored at the stage boundary; an
    already captured graph keeps the kernels selected during its capture.
    """
    requested_mode = _stage_tf32_mode()
    mode = requested_mode if _stage_tf32_applies(dataset) else "none"
    ddim_prefix_steps = stage_tf32_ddim_prefix_steps()
    if stage not in {"decision", "simulation"}:
        raise ValueError(f"Unknown generation stage {stage!r}")

    old_cudnn = torch.backends.cudnn.allow_tf32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_benchmark = torch.backends.cudnn.benchmark
    selected_stage, _, selected_backend = mode.partition("-")
    enabled = (mode != "none" and stage == selected_stage and
               ddim_prefix_steps == 0)
    requested_benchmark_stage = _stage_cudnn_benchmark_mode()
    benchmark_stage = (requested_benchmark_stage
                       if _stage_cudnn_benchmark_applies(dataset) else "none")
    benchmark_key = (dataset, stage)
    benchmark_once = (stage == benchmark_stage and
                      benchmark_key not in _STAGE_CUDNN_BENCHMARK_DONE)
    try:
        if mode != "none":
            # The experiment is intentionally narrow: one backend in one
            # stage.  Starting from global FP32 avoids accidental combinations.
            if old_cudnn or old_matmul:
                raise ValueError(
                    "UNIFOLM_STAGE_TF32 requires global FP32 math mode")
            torch.backends.cudnn.allow_tf32 = (
                enabled and selected_backend == "cudnn")
            torch.backends.cuda.matmul.allow_tf32 = (
                enabled and selected_backend == "matmul")
        if benchmark_stage != "none":
            if old_benchmark:
                raise ValueError(
                    "UNIFOLM_STAGE_CUDNN_BENCHMARK requires global cuDNN "
                    "benchmark off")
            torch.backends.cudnn.benchmark = benchmark_once
        yield
    finally:
        if benchmark_once:
            _STAGE_CUDNN_BENCHMARK_DONE.add(benchmark_key)
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.benchmark = old_benchmark


@contextmanager
def ddim_step_math(stage: str, dataset: str, iteration: int):
    """Apply stage TF32 only to a leading prefix of DDIM iterations.

    A zero prefix retains the whole-stage policy implemented by
    :func:`generation_math`. CUDA Graph cache keys include these backend flags,
    so FP32 and TF32 iterations replay graphs captured with matching kernels.
    """
    prefix_steps = stage_tf32_ddim_prefix_steps()
    if prefix_steps == 0:
        yield
        return
    if stage not in {"decision", "simulation"}:
        raise ValueError(f"Unknown generation stage {stage!r}")
    mode = _stage_tf32_mode() if _stage_tf32_applies(dataset) else "none"
    selected_stage, _, selected_backend = mode.partition("-")
    enabled = (mode != "none" and stage == selected_stage and
               iteration < prefix_steps)
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = (
            enabled and selected_backend == "cudnn")
        torch.backends.cuda.matmul.allow_tf32 = (
            enabled and selected_backend == "matmul")
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.backends.cuda.matmul.allow_tf32 = old_matmul


def configure_denoiser(model, mode: str, dataset: str) -> None:
    full_fp16_mode = os.environ.get("UNIFOLM_FULL_DENOISER_FP16", "0")
    full_fp16_enabled = full_denoiser_fp16_enabled(full_fp16_mode, dataset)
    backbone_amp = os.environ.get("UNIFOLM_BACKBONE_AMP", "0")
    if backbone_amp not in {"0", "fp16"}:
        raise ValueError("UNIFOLM_BACKBONE_AMP must be 0 or fp16")
    if backbone_amp != "0" and (full_fp16_enabled or mode != "native" or
            os.environ.get("UNIFOLM_TF32X3_LINEARS", "0") != "0"):
        raise ValueError("Backbone AMP requires native FP32 master weights")
    full_fp16_norm_dtype = os.environ.get(
        "UNIFOLM_FULL_DENOISER_NORM_DTYPE", "fp32")
    if full_fp16_norm_dtype not in {"fp32", "fp16"}:
        raise ValueError(
            "UNIFOLM_FULL_DENOISER_NORM_DTYPE must be fp32 or fp16")
    if (full_fp16_norm_dtype == "fp16" and not full_fp16_enabled and
            full_fp16_mode not in {"non-v2", "validated"}):
        raise ValueError(
            "FP16 denoiser norms require UNIFOLM_FULL_DENOISER_FP16")
    full_fp16_norm_count = 0
    fp32_feedback = os.environ.get("UNIFOLM_FP32_FEEDBACK_HEADS", "0")
    if fp32_feedback not in {"0", "1"}:
        raise ValueError("UNIFOLM_FP32_FEEDBACK_HEADS must be 0 or 1")
    if fp32_feedback == "1" and not full_fp16_enabled:
        raise ValueError("FP32 feedback experiment requires FP16 backbone")
    fp32_video_boundaries = os.environ.get("UNIFOLM_FP32_VIDEO_BOUNDARIES", "0")
    if fp32_video_boundaries not in {"0", "output", "output+embeddings"}:
        raise ValueError("UNIFOLM_FP32_VIDEO_BOUNDARIES must be 0, output, or output+embeddings")
    if fp32_video_boundaries != "0" and fp32_feedback != "1":
        raise ValueError("FP32 video boundaries require FP32 feedback with FP16 backbone")
    fp32_temporal_blocks = os.environ.get("UNIFOLM_FP32_TEMPORAL_BLOCKS", "0")
    if fp32_temporal_blocks not in {"0", "all", "c640"}:
        raise ValueError("UNIFOLM_FP32_TEMPORAL_BLOCKS must be 0, all, or c640")
    if fp32_temporal_blocks != "0" and fp32_feedback != "1":
        raise ValueError("FP32 temporal blocks require FP32 feedback with FP16 backbone")
    protected_feedback_ids = set()
    fp32_ddim_video = os.environ.get("UNIFOLM_FP32_DDIM_VIDEO", "0")
    if fp32_ddim_video not in {"0", "1"}:
        raise ValueError("UNIFOLM_FP32_DDIM_VIDEO must be 0 or 1")
    if fp32_ddim_video == "1" and not full_fp16_enabled:
        raise ValueError("FP32 DDIM video experiment requires FP16 backbone")
    model.model.diffusion_model._fp32_ddim_video = fp32_ddim_video == "1"
    print(f"FP32_DDIM_VIDEO requested={fp32_ddim_video}", flush=True)
    if full_fp16_enabled:
        diffusion = model.model.diffusion_model
        if fp32_feedback == "1":
            from unifolm_wma.utils.mixed_precision_feedback import convert_half_with_fp32_feedback
            protected_feedback_ids = convert_half_with_fp32_feedback(
                diffusion, protect_video_output=fp32_video_boundaries != "0",
                protect_condition_embeddings=fp32_video_boundaries == "output+embeddings",
                protected_temporal_channels={"0": (), "all": (320, 640, 1280),
                                             "c640": (640,)}[fp32_temporal_blocks])
        else:
            diffusion.half()
        diffusion.dtype = torch.float16
        from unifolm_wma.utils.basics import GroupNormSpecific
        for module in diffusion.modules():
            if isinstance(module, GroupNormSpecific):
                if id(module) in protected_feedback_ids:
                    module._inference_dtype = torch.float32
                    continue
                module._inference_dtype = (
                    torch.float16 if full_fp16_norm_dtype == "fp16"
                    else torch.float32)
                if full_fp16_norm_dtype == "fp32":
                    # The precision-preserving default promotes activations,
                    # so affine parameters must remain FP32 too.
                    module.float()
                full_fp16_norm_count += 1
    print(f"FP32_FEEDBACK_HEADS requested={fp32_feedback} "
          f"protected_modules={len(protected_feedback_ids)}", flush=True)
    print(f"FP32_VIDEO_BOUNDARIES requested={fp32_video_boundaries}", flush=True)
    print(f"FULL_DENOISER_FP16 requested={full_fp16_mode} "
          f"enabled={str(full_fp16_enabled).lower()}", flush=True)
    print(f"FULL_DENOISER_NORM_DTYPE requested={full_fp16_norm_dtype} "
          f"modules={full_fp16_norm_count}", flush=True)
    if mode not in {"native", "tf32x3", "selective-tf32x3"}:
        raise ValueError(f"Unknown UNIFOLM_DENOISER_MODE={mode!r}")
    enabled = mode == "tf32x3" or (mode == "selective-tf32x3" and
                                  dataset == "unitree_z1_dual_arm_stackbox_v2")
    tf32x3_linears = os.environ.get("UNIFOLM_TF32X3_LINEARS", "0")
    if tf32x3_linears not in {"0", "1", "triton"}:
        raise ValueError("UNIFOLM_TF32X3_LINEARS must be 0, 1, or triton")
    compensated_linear_backend = os.environ.get("UNIFOLM_COMPENSATED_LINEAR_BACKEND", "tf32")
    if compensated_linear_backend not in {"tf32", "scaled-fp16"}:
        raise ValueError("UNIFOLM_COMPENSATED_LINEAR_BACKEND must be tf32 or scaled-fp16")
    if compensated_linear_backend != "tf32" and tf32x3_linears != "triton":
        raise ValueError("Scaled FP16 products require Triton compensated linears")
    packed_half_weights = os.environ.get('UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE', '0')
    if packed_half_weights not in {'0', '1'}:
        raise ValueError('UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE must be 0 or 1')
    if packed_half_weights == '1' and compensated_linear_backend != 'scaled-fp16':
        raise ValueError('Packed weight cache requires scaled FP16 compensated linears')
    linear_group_m = int(os.environ.get('UNIFOLM_COMPENSATED_LINEAR_GROUP_M', '0'))
    if linear_group_m not in {0, 4, 8, 16, 32} or (
            linear_group_m and compensated_linear_backend != 'scaled-fp16'):
        raise ValueError('Grouped scheduling requires scaled FP16 linears and group 0/4/8/16/32')
    print(f'COMPENSATED_LINEAR_GROUP_M requested={linear_group_m} arithmetic_unchanged=true', flush=True)
    half_linear_block_k = int(os.environ.get('UNIFOLM_COMPENSATED_LINEAR_BLOCK_K', '32'))
    if half_linear_block_k not in {32, 64, 128} or (
            half_linear_block_k != 32 and compensated_linear_backend != 'scaled-fp16'):
        raise ValueError('Reduction tile must be 32/64/128 and larger tiles require scaled FP16 backend')
    print(f'COMPENSATED_LINEAR_BLOCK_K requested={half_linear_block_k} '
          'accumulation=fp32 output=fp32', flush=True)
    input_pack = os.environ.get('UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK', '0')
    if input_pack not in {'0', 'wide'}:
        raise ValueError('UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK must be 0 or wide')
    if input_pack != '0' and (compensated_linear_backend != 'scaled-fp16' or packed_half_weights != '1'):
        raise ValueError('Input packing requires cached compensated Half weights')
    print(f'COMPENSATED_LINEAR_INPUT_PACK requested={input_pack} '
          'minimum_output_channels=1280 refreshed_each_call=true packing_in_timing=true', flush=True)
    narrow_output = os.environ.get("UNIFOLM_TF32X3_NARROW_OUTPUT", "0")
    if narrow_output not in {"0", "1"}:
        raise ValueError("UNIFOLM_TF32X3_NARROW_OUTPUT must be 0 or 1")
    if narrow_output == "1" and tf32x3_linears != "triton":
        raise ValueError("Narrow output experiment requires Triton compensated linears")
    narrow_ffn = os.environ.get("UNIFOLM_TF32X3_NARROW_FFN", "0")
    if narrow_ffn not in {"0", "1"}:
        raise ValueError("UNIFOLM_TF32X3_NARROW_FFN must be 0 or 1")
    if narrow_ffn == "1" and tf32x3_linears != "triton":
        raise ValueError("Narrow FFN experiment requires Triton compensated linears")
    if full_fp16_enabled and tf32x3_linears != "0":
        raise ValueError("TF32x3 linears require FP32 denoiser parameters")
    if enabled or tf32x3_linears != "0":
        if torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("TF32x3 requires FP32 outside the selected operators")
        if tf32x3_linears == "triton":
            if enabled:
                raise ValueError("Triton compensated linears require native convolutions")
            from unifolm_wma.utils.triton_compensated_linear import install_compensated_linears
            counts = install_compensated_linears(
                model.model.diffusion_model, narrow_output=narrow_output == "1",
                narrow_ffn=narrow_ffn == "1", backend=compensated_linear_backend,
                cache_half_weights=packed_half_weights == '1', group_m=linear_group_m, block_k=half_linear_block_k, pack_inputs=input_pack == 'wide')
        else:
            from unifolm_wma.utils.tf32x3 import install_tf32x3
            counts = install_tf32x3(model.model.diffusion_model,
                                    convolutions=enabled)
    else:
        counts = {}
    initial_decision_mode = os.environ.get('UNIFOLM_INITIAL_DECISION_LINEARS', '0')
    if initial_decision_mode not in {'0','1'}:
        raise ValueError('UNIFOLM_INITIAL_DECISION_LINEARS must be 0 or 1')
    initial_decision_count = 0
    if initial_decision_mode == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or
                stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('Initial decision linears require V2 FP32 and full decision-cudnn')
        from .initial_decision_linears import install_initial_decision_linears
        initial_decision_count = install_initial_decision_linears(model.model.diffusion_model)
    print(f'INITIAL_DECISION_LINEARS requested={initial_decision_mode} '
          f'modules={initial_decision_count} BK=32 world=native_fp32 '
          'master_output=fp32 weights=first_timed_warm', flush=True)
    vendor_ffn_mode = os.environ.get('UNIFOLM_DECISION_VENDOR_FFN','0')
    if vendor_ffn_mode not in {'0','full','chunked'}:
        raise ValueError('UNIFOLM_DECISION_VENDOR_FFN must be 0,full,or chunked')
    vendor_ffn_count = 0
    if vendor_ffn_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                compensated_linear_backend!='scaled-fp16' or packed_half_weights!='1'):
            raise ValueError('Vendor FFN requires accepted V2 FP32 and decision-cudnn policy')
        from .decision_vendor_ffn import install_decision_vendor_ffns
        vendor_ffn_count = install_decision_vendor_ffns(model.model.diffusion_model,vendor_ffn_mode)
    print(f'DECISION_VENDOR_FFN requested={vendor_ffn_mode} modules={vendor_ffn_count} '
          'scope=decision_ffn_output640,1280 world=accepted_shortBK64 '
          'master_bias_accum_output=fp32 weights=first_timed_warm',flush=True)
    print(f"DENOISER_MODE requested={mode} case={dataset} wrapped={counts}", flush=True)
    print(f"TF32X3_LINEARS requested={tf32x3_linears} "
          f"convolutions={str(enabled).lower()} wrapped={counts}", flush=True)
    print(f"COMPENSATED_LINEAR_BACKEND requested={compensated_linear_backend} "
          "parameter_dtype=fp32 output_dtype=fp32", flush=True)
    print(f'COMPENSATED_LINEAR_WEIGHT_CACHE requested={packed_half_weights} '
          'warmup_in_timing=true', flush=True)
    print(f"TF32X3_NARROW_OUTPUT requested={narrow_output} rows=40960 "
          "in=320,1280 out=320", flush=True)
    print(f"TF32X3_NARROW_FFN requested={narrow_ffn} rows=40960 "
          "in=320 out=2560", flush=True)
    compensated_geglu = os.environ.get("UNIFOLM_COMPENSATED_GEGLU", "0")
    if compensated_geglu not in {"0", "narrow"}:
        raise ValueError("UNIFOLM_COMPENSATED_GEGLU must be 0 or narrow")
    compensated_geglu_count = 0
    if compensated_geglu == "narrow":
        if full_fp16_enabled or narrow_ffn == "1" or tf32x3_linears != "triton":
            raise ValueError("Compensated GEGLU requires FP32 Triton linears without narrow FFN")
        from unifolm_wma.utils.triton_compensated_geglu import install_compensated_geglus
        compensated_geglu_count = install_compensated_geglus(model.model.diffusion_model)
    print(f"COMPENSATED_GEGLU requested={compensated_geglu} "
          f"modules={compensated_geglu_count} rows=40960 in=320 out=1280", flush=True)
    prepacked_geglu_mode = os.environ.get('UNIFOLM_PREPACKED_GEGLU', '0')
    if prepacked_geglu_mode not in {'0','1'}:
        raise ValueError('UNIFOLM_PREPACKED_GEGLU must be 0 or 1')
    prepacked_geglu_count = 0
    if prepacked_geglu_mode == '1':
        if (full_fp16_enabled or tf32x3_linears != 'triton' or
                compensated_linear_backend != 'scaled-fp16' or
                packed_half_weights != '1' or input_pack != 'wide' or
                half_linear_block_k != 64 or linear_group_m != 32):
            raise ValueError('Wide GEGLU requires accepted prepacked3Half BK64 group32 FP32 policy')
        from .triton_prepacked_geglu import install_prepacked_geglus
        prepacked_geglu_count = install_prepacked_geglus(model.model.diffusion_model)
    print(f'PREPACKED_GEGLU requested={prepacked_geglu_mode} '
          f'modules={prepacked_geglu_count} channels=640,1280 BK=64 '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'input_parts=refreshed_each_call',flush=True)
    decision_half_geglu_mode = os.environ.get('UNIFOLM_DECISION_HALF_GEGLU', '0')
    if decision_half_geglu_mode not in {'0', '1'}:
        raise ValueError('UNIFOLM_DECISION_HALF_GEGLU must be 0 or 1')
    decision_half_geglu_count = 0
    if decision_half_geglu_mode == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('Decision Half GEGLU requires V2 FP32 decision-cudnn stage policy')
        from .decision_half_geglu import install_decision_half_geglus
        decision_half_geglu_count = install_decision_half_geglus(model.model.diffusion_model)
    print(f'DECISION_HALF_GEGLU requested={decision_half_geglu_mode} '
          f'modules={decision_half_geglu_count} channels=320 BK=32 group=16 '
          'world=accepted_3TF32 master_bias_accum_output=fp32', flush=True)
    fused_qkv_mode = os.environ.get("UNIFOLM_FUSED_SELF_QKV", "0")
    if fused_qkv_mode not in {"0", "1", "selective"}:
        raise ValueError(
            "UNIFOLM_FUSED_SELF_QKV must be 0, 1, or selective")
    fused_qkv_enabled = (
        fused_qkv_mode == "1" or
        (fused_qkv_mode == "selective" and
         dataset in FUSED_QKV_CANDIDATE_CASES))
    from unifolm_wma.modules.attention import CrossAttention
    fused_qkv_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is CrossAttention:
            module._fused_self_qkv_enabled = fused_qkv_enabled
            fused_qkv_count += 1
    print(f"FUSED_SELF_QKV requested={fused_qkv_mode} "
          f"enabled={str(fused_qkv_enabled).lower()} "
          f"modules={fused_qkv_count}", flush=True)
    compensated_qkv = os.environ.get('UNIFOLM_COMPENSATED_SELF_QKV', '0')
    if compensated_qkv not in {'0', '1'}:
        raise ValueError('UNIFOLM_COMPENSATED_SELF_QKV must be 0 or 1')
    compensated_qkv_count = 0
    if compensated_qkv == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or fused_qkv_enabled or
                compensated_linear_backend != 'scaled-fp16' or
                os.environ.get('UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK', '0') != 'wide' or
                os.environ.get('UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE', '0') != '1' or
                os.environ.get('UNIFOLM_COMPENSATED_LINEAR_GROUP_M', '0') != '32' or
                os.environ.get('UNIFOLM_COMPENSATED_LINEAR_BLOCK_K', '32') != '64'):
            raise ValueError('Compensated QKV requires V2 FP32 prepacked BK64/group32 base')
        from unifolm_wma.utils.compensated_qkv import install_compensated_qkv
        compensated_qkv_count = install_compensated_qkv(model.model.diffusion_model)
    print(f'COMPENSATED_SELF_QKV requested={compensated_qkv} '
          f'modules={compensated_qkv_count} output_layout=three_contiguous_fp32 '
          'input_conversion_in_timing=true', flush=True)
    cross_attention_mode = os.environ.get(
        "UNIFOLM_CROSS_ATTENTION_IMPL", "xformers")
    if cross_attention_mode not in {
            "xformers", "torch", "selective", "small-k-tf32"}:
        raise ValueError(
            "UNIFOLM_CROSS_ATTENTION_IMPL must be xformers, torch, "
            "selective, or small-k-tf32")
    torch_cross_attention_enabled = (
        cross_attention_mode == "torch" or
        (cross_attention_mode == "selective" and
         dataset in CROSS_ATTENTION_CANDIDATE_CASES))
    attention_layout = os.environ.get("UNIFOLM_ATTENTION_LAYOUT", "packed")
    if attention_layout not in {"packed", "bmhk"}:
        raise ValueError("UNIFOLM_ATTENTION_LAYOUT must be packed or bmhk")
    if attention_layout == "bmhk" and cross_attention_mode != "xformers":
        raise ValueError("BMHK attention layout requires xformers attention")
    fused_attention_mix = os.environ.get("UNIFOLM_FUSED_ATTENTION_MIX", "0")
    if fused_attention_mix not in {"0", "1"}:
        raise ValueError("UNIFOLM_FUSED_ATTENTION_MIX must be 0 or 1")
    cross_attention_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is CrossAttention:
            module._inference_attention_layout = attention_layout
            module._inference_fused_mix = fused_attention_mix == "1"
            module._torch_cross_attention_enabled = torch_cross_attention_enabled
            module._torch_small_k_enabled = (
                cross_attention_mode == "small-k-tf32")
            cross_attention_count += 1
    print(f"CROSS_ATTENTION_IMPL requested={cross_attention_mode} "
          f"effective={'torch' if torch_cross_attention_enabled else cross_attention_mode} "
          f"modules={cross_attention_count}", flush=True)
    print(f"ATTENTION_LAYOUT requested={attention_layout} "
          f"modules={cross_attention_count}", flush=True)
    print(f"FUSED_ATTENTION_MIX requested={fused_attention_mix} "
          f"modules={cross_attention_count}", flush=True)
    attention_dtype_mode = os.environ.get("UNIFOLM_ATTENTION_DTYPE", "fp32")
    attention_modes = {
        "fp32": (torch.float32, False),
        "bf16": (torch.bfloat16, False),
        "fp16": (torch.float16, False),
        "bf16-self": (torch.bfloat16, True),
        "fp16-self": (torch.float16, True),
    }
    if attention_dtype_mode == "validated":
        attention_dtype = (torch.float16 if dataset in VALIDATED_FP16_CASES
                           else torch.float32)
        attention_self_only = False
    elif attention_dtype_mode in attention_modes:
        attention_dtype, attention_self_only = attention_modes[
            attention_dtype_mode]
    else:
        raise ValueError(
            "UNIFOLM_ATTENTION_DTYPE must be fp32, bf16, fp16, "
            "bf16-self, fp16-self, or validated")
    attention_dtype_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is CrossAttention:
            module._inference_attention_dtype = attention_dtype
            module._inference_attention_self_only = attention_self_only
            attention_dtype_count += 1
    print(f"ATTENTION_DTYPE requested={attention_dtype_mode} "
          f"effective={str(attention_dtype).removeprefix('torch.')} "
          f"modules={attention_dtype_count}", flush=True)
    projection_dtype_mode = os.environ.get(
        "UNIFOLM_ATTENTION_PROJECTION_DTYPE", "fp32")
    projection_dtypes = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if projection_dtype_mode == "validated":
        projection_dtype = (torch.float16 if dataset in VALIDATED_FP16_CASES
                            else torch.float32)
    elif projection_dtype_mode in projection_dtypes:
        projection_dtype = projection_dtypes[projection_dtype_mode]
    else:
        raise ValueError(
            "UNIFOLM_ATTENTION_PROJECTION_DTYPE must be fp32, bf16, fp16, "
            "or validated")
    projection_dtype_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is CrossAttention:
            module._inference_projection_dtype = projection_dtype
            projection_dtype_count += 1
    print(f"ATTENTION_PROJECTION_DTYPE requested={projection_dtype_mode} "
          f"effective={str(projection_dtype).removeprefix('torch.')} "
          f"modules={projection_dtype_count}", flush=True)
    compensated_attention = os.environ.get('UNIFOLM_COMPENSATED_SPATIAL_ATTENTION', '0')
    if compensated_attention not in {'0', 'large'}:
        raise ValueError('UNIFOLM_COMPENSATED_SPATIAL_ATTENTION must be 0 or large')
    compensated_attention_enabled = (
        compensated_attention == 'large' and mode == 'native' and
        not full_fp16_enabled and os.environ.get('UNIFOLM_BACKBONE_AMP', '0') == '0' and
        attention_dtype == torch.float32 and projection_dtype == torch.float32)
    compensated_attention_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is CrossAttention:
            module._compensated_spatial_attention = compensated_attention_enabled
            compensated_attention_count += int(compensated_attention_enabled)
    print(f'COMPENSATED_SPATIAL_ATTENTION requested={compensated_attention} '
          f'enabled={compensated_attention_enabled} '
          f'modules={compensated_attention_count} block=64x64', flush=True)
    ffn_dtype_mode = os.environ.get("UNIFOLM_FFN_DTYPE", "fp32")
    ffn_dtypes = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if ffn_dtype_mode not in ffn_dtypes:
        raise ValueError("UNIFOLM_FFN_DTYPE must be fp32, bf16, or fp16")
    from unifolm_wma.modules.attention import FeedForward
    ffn_dtype_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is FeedForward:
            module._inference_dtype = ffn_dtypes[ffn_dtype_mode]
            ffn_dtype_count += 1
    print(f"FFN_DTYPE requested={ffn_dtype_mode} modules={ffn_dtype_count}",
          flush=True)
    fused_geglu = os.environ.get("UNIFOLM_FUSED_GEGLU", "0")
    if fused_geglu not in {"0", "1"}:
        raise ValueError("UNIFOLM_FUSED_GEGLU must be 0 or 1")
    from unifolm_wma.modules.attention import GEGLU
    fused_geglu_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is GEGLU:
            module._inference_fused_geglu = fused_geglu == "1"
            fused_geglu_count += 1
    print(f"FUSED_GEGLU requested={fused_geglu} modules={fused_geglu_count}",
          flush=True)
    nvrtc_geglu = os.environ.get("UNIFOLM_NVRTC_GEGLU", "0")
    if nvrtc_geglu not in {"0", "fmad", "no-fmad"}:
        raise ValueError("UNIFOLM_NVRTC_GEGLU must be 0, fmad, or no-fmad")
    nvrtc_geglu_vector = os.environ.get("UNIFOLM_NVRTC_GEGLU_VECTOR", "0")
    if nvrtc_geglu_vector not in {"0", "1"}:
        raise ValueError("UNIFOLM_NVRTC_GEGLU_VECTOR must be 0 or 1")
    if nvrtc_geglu_vector == "1" and nvrtc_geglu == "0":
        raise ValueError("NVRTC GEGLU vectorization requires the NVRTC backend")
    nvrtc_geglu_count = 0
    if nvrtc_geglu != "0":
        if full_fp16_enabled or ffn_dtype_mode != "fp32":
            raise ValueError("NVRTC GEGLU requires FP32 FFN")
        # A narrow compensated forward owns its eligible calls and bypasses
        # activate(). NVRTC handles the other GEGLUs and the native fallbacks.
        from unifolm_wma.utils.nvrtc_geglu import install_nvrtc_geglus
        nvrtc_geglu_count = install_nvrtc_geglus(
            model.model.diffusion_model, fmad=nvrtc_geglu == "fmad",
            vectorized=nvrtc_geglu_vector == "1")
    print(f"NVRTC_GEGLU requested={nvrtc_geglu} modules={nvrtc_geglu_count}", flush=True)
    print(f"NVRTC_GEGLU_VECTOR requested={nvrtc_geglu_vector} block=128 items=1", flush=True)
    fused_norm = os.environ.get("UNIFOLM_FUSED_RESBLOCK_NORM", "0")
    if fused_norm not in {"0", "1", "selective"}:
        raise ValueError(
            "UNIFOLM_FUSED_RESBLOCK_NORM must be 0, 1, or selective")
    fused_norm_enabled = (
        fused_norm == "1" or
        (fused_norm == "selective" and dataset in FUSED_QKV_CANDIDATE_CASES))
    from unifolm_wma.modules.networks.wma_model import ResBlock
    fused_norm_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is ResBlock:
            module._fused_groupnorm_silu = fused_norm_enabled
            fused_norm_count += 1
    print(f"FUSED_RESBLOCK_NORM requested={fused_norm} "
          f"enabled={str(fused_norm_enabled).lower()} "
          f"modules={fused_norm_count}", flush=True)
    framewise_norm = os.environ.get("UNIFOLM_FRAMEWISE_TEMPORAL_NORM", "0")
    if framewise_norm not in {"0", "1", "selective"}:
        raise ValueError(
            "UNIFOLM_FRAMEWISE_TEMPORAL_NORM must be 0, 1, or selective")
    framewise_norm_enabled = (
        framewise_norm == "1" or
        (framewise_norm == "selective" and
         dataset in FUSED_QKV_CANDIDATE_CASES))
    from unifolm_wma.modules.networks.wma_model import TemporalConvBlock
    temporal_conv_dtype_mode = os.environ.get(
        "UNIFOLM_TEMPORAL_CONV_DTYPE", "fp32")
    if temporal_conv_dtype_mode not in {
            "fp32", "fp16", "v2-fp16", "v2-c640-fp16"}:
        raise ValueError(
            "UNIFOLM_TEMPORAL_CONV_DTYPE must be fp32, fp16, v2-fp16, "
            "or v2-c640-fp16")
    temporal_conv_dtype_enabled = (
        not full_fp16_enabled and
        (temporal_conv_dtype_mode == "fp16" or
         (temporal_conv_dtype_mode in {"v2-fp16", "v2-c640-fp16"} and
          dataset == STAGE_TF32_CASE)))
    temporal_conv_dtype_count = 0
    if temporal_conv_dtype_enabled:
        for module in model.model.diffusion_model.modules():
            if (type(module) is TemporalConvBlock and
                    (temporal_conv_dtype_mode != "v2-c640-fp16" or
                     module.in_channels == 640)):
                module._inference_conv_dtype = torch.float16
                for block in (module.conv1, module.conv2, module.conv3,
                              module.conv4):
                    block[-1].to(torch.float16)
                temporal_conv_dtype_count += 1
    print(f"TEMPORAL_CONV_DTYPE requested={temporal_conv_dtype_mode} "
          f"enabled={str(temporal_conv_dtype_enabled).lower()} "
          f"modules={temporal_conv_dtype_count}", flush=True)
    temporal_conv_impl = os.environ.get("UNIFOLM_TEMPORAL_CONV_IMPL", "native")
    if temporal_conv_impl not in {"native", "conv2d", "fp32-conv2d"}:
        raise ValueError(
            "UNIFOLM_TEMPORAL_CONV_IMPL must be native, conv2d, or fp32-conv2d")
    compensated_temporal_mode = os.environ.get('UNIFOLM_COMPENSATED_TEMPORAL_CONV', '0')
    if compensated_temporal_mode not in {'0','1'}:
        raise ValueError('UNIFOLM_COMPENSATED_TEMPORAL_CONV must be 0 or 1')
    if compensated_temporal_mode == '1' and temporal_conv_impl != 'fp32-conv2d':
        raise ValueError('Compensated temporal convolution requires accepted fp32-conv2d')
    compensated_temporal_count = 0
    temporal_conv2d_count = 0
    if temporal_conv_impl in {"conv2d", "fp32-conv2d"}:
        if enabled:
            raise ValueError("Temporal Conv2d cannot be combined with TF32x3")
        from unifolm_wma.utils.temporal_conv2d import make_temporal_conv2d_forward
        for module in model.model.diffusion_model.modules():
            if type(module) is TemporalConvBlock:
                for block in (module.conv1, module.conv2, module.conv3,
                              module.conv4):
                    conv = block[-1]
                    conv.forward = make_temporal_conv2d_forward(
                        conv, preserve_tf32=temporal_conv_impl == "fp32-conv2d")
                    if compensated_temporal_mode == '1':
                        from .triton_compensated_temporal_conv import make_compensated_temporal_forward
                        conv.forward = make_compensated_temporal_forward(conv)
                        compensated_temporal_count += 1
                    temporal_conv2d_count += 1
    print(f'COMPENSATED_TEMPORAL_CONV requested={compensated_temporal_mode} '
          f'modules={compensated_temporal_count} tf32_stage=native '
          'master_accum_output=fp32 static_weights=first_timed_warm BK=64',flush=True)
    print(f"TEMPORAL_CONV_IMPL requested={temporal_conv_impl} "
          f"modules={temporal_conv2d_count}", flush=True)
    framewise_norm_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is TemporalConvBlock:
            module._fused_framewise_norm = framewise_norm_enabled
            framewise_norm_count += 1
    print(f"FRAMEWISE_TEMPORAL_NORM requested={framewise_norm} "
          f"enabled={str(framewise_norm_enabled).lower()} "
          f"modules={framewise_norm_count}", flush=True)
    exact_temporal_norm = os.environ.get("UNIFOLM_FUSED_TEMPORAL_NORM", "0")
    if exact_temporal_norm not in {"0", "1", "selective"}:
        raise ValueError(
            "UNIFOLM_FUSED_TEMPORAL_NORM must be 0, 1, or selective")
    exact_temporal_norm_enabled = (
        exact_temporal_norm == "1" or
        (exact_temporal_norm == "selective" and
         dataset in FUSED_QKV_CANDIDATE_CASES))
    if framewise_norm_enabled and exact_temporal_norm_enabled:
        raise ValueError(
            "Framewise and exact fused temporal norms are mutually exclusive")
    exact_temporal_norm_count = 0
    for module in model.model.diffusion_model.modules():
        if type(module) is TemporalConvBlock:
            module._fused_exact_temporal_norm = exact_temporal_norm_enabled
            exact_temporal_norm_count += 1
    print(f"FUSED_TEMPORAL_NORM requested={exact_temporal_norm} "
          f"enabled={str(exact_temporal_norm_enabled).lower()} "
          f"modules={exact_temporal_norm_count}", flush=True)
    channels_last = os.environ.get("UNIFOLM_CHANNELS_LAST", "0")
    if channels_last not in {"0", "1", "selective"}:
        raise ValueError("UNIFOLM_CHANNELS_LAST must be 0, 1, or selective")
    channels_last_enabled = (
        channels_last == "1" or
        (channels_last == "selective" and
         dataset in FUSED_QKV_CANDIDATE_CASES))
    channels_last_count = 0
    if channels_last_enabled:
        diffusion = model.model.diffusion_model
        diffusion._channels_last_2d_enabled = True
        from unifolm_wma.modules.attention import (SpatialTransformer,
                                                   TemporalTransformer)
        for module in diffusion.modules():
            if isinstance(module, SpatialTransformer):
                module._channels_last_output = True
                channels_last_count += 1
            elif isinstance(module, TemporalTransformer):
                module._channels_last_output = True
                channels_last_count += 1
            elif isinstance(module, torch.nn.Conv2d):
                module.weight.data = module.weight.data.contiguous(
                    memory_format=torch.channels_last)
            elif isinstance(module, torch.nn.Conv3d):
                module.weight.data = module.weight.data.contiguous(
                    memory_format=torch.channels_last_3d)
    print(f"CHANNELS_LAST requested={channels_last} "
          f"enabled={str(channels_last_enabled).lower()} "
          f"transformers={channels_last_count}", flush=True)
    split_norm = os.environ.get("UNIFOLM_SPLIT_GROUP_NORM", "0")
    compensated_spatial = os.environ.get("UNIFOLM_COMPENSATED_SPATIAL_CONV", "0")
    if compensated_spatial not in {"0", "small"}:
        raise ValueError("UNIFOLM_COMPENSATED_SPATIAL_CONV must be 0 or small")
    compensated_spatial_count = 0
    if compensated_spatial == "small":
        from unifolm_wma.utils.compensated_spatial_conv import install_compensated_spatial_convs
        compensated_spatial_count = install_compensated_spatial_convs(model.model.diffusion_model)
    print(f"COMPENSATED_SPATIAL_CONV requested={compensated_spatial} "
          f"modules={compensated_spatial_count} shape=5x8 tf32_stage=native", flush=True)
    winograd_conv = os.environ.get("UNIFOLM_WINOGRAD_CONV", "0")
    if winograd_conv not in {"0", "small", "small1280", "small2560", "medium", "smallmedium"}:
        raise ValueError("Invalid UNIFOLM_WINOGRAD_CONV mode")
    winograd_count = 0
    if winograd_conv != "0":
        if compensated_spatial != "0" or full_fp16_enabled:
            raise ValueError("Winograd requires FP32 backbone and no other spatial replacement")
        from unifolm_wma.utils.triton_winograd import install_winograd_convs
        inputs = {"small1280": (1280,), "small2560": (2560,)}
        spatial_shapes = {"medium": ((10, 16),), "smallmedium": ((5, 8), (10, 16))}
        winograd_count = install_winograd_convs(
            model.model.diffusion_model, inputs=inputs.get(winograd_conv, (1280, 2560)),
            spatial_shapes=spatial_shapes.get(winograd_conv, ((5, 8),)))
    print(f"WINOGRAD_CONV requested={winograd_conv} modules={winograd_count} "
          "batch=16 tf32_stage=native filter_cache=timed_warmup", flush=True)
    winograd_f2 = os.environ.get('UNIFOLM_WINOGRAD_F2', '0')
    if winograd_f2 not in {'0', 'high', 'medium', 'deep'}:
        raise ValueError('UNIFOLM_WINOGRAD_F2 must be 0, high, medium, or deep')
    winograd_f2_count = 0
    if winograd_f2 != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or mode != 'native' or
                winograd_conv != '0' or compensated_spatial != '0'):
            raise ValueError('F2 requires V2 FP32 native convolutions and no other spatial replacement')
        from unifolm_wma.utils.triton_winograd_f2 import install_f2_convolutions
        winograd_f2_count = install_f2_convolutions(model.model.diffusion_model, scope=winograd_f2)
    print(f'WINOGRAD_F2 requested={winograd_f2} modules={winograd_f2_count} '
          f'spatial_scope={winograd_f2} master=fp32 transforms=fp32 '
          'component_accum=fp32 filter_cache=first_timed_warm', flush=True)
    fused_temporal_attention = os.environ.get("UNIFOLM_FUSED_TEMPORAL_ATTENTION", "0")
    if fused_temporal_attention not in {"0", "1"}:
        raise ValueError("UNIFOLM_FUSED_TEMPORAL_ATTENTION must be 0 or 1")
    for module in model.model.diffusion_model.modules():
        if isinstance(module, CrossAttention):
            module._inference_fused_temporal_attention = fused_temporal_attention == "1"
    print(f"FUSED_TEMPORAL_ATTENTION requested={fused_temporal_attention}",flush=True)
    fused_temporal_fp32 = os.environ.get("UNIFOLM_FUSED_TEMPORAL_ATTENTION_FP32", "0")
    if fused_temporal_fp32 not in {"0", "1"}:
        raise ValueError("UNIFOLM_FUSED_TEMPORAL_ATTENTION_FP32 must be 0 or 1")
    native_fp32 = os.environ.get("UNIFOLM_NATIVE_FP32_TEMPORAL_ATTENTION", "0")
    if native_fp32 not in {"0", "1"}:
        raise ValueError("UNIFOLM_NATIVE_FP32_TEMPORAL_ATTENTION must be 0 or 1")
    if native_fp32 == "1" and fused_temporal_fp32 != "1":
        raise ValueError("Native FP32 attention requires FP32 fused attention enabled")
    for module in model.model.diffusion_model.modules():
        if isinstance(module, CrossAttention):
            module._inference_fused_temporal_attention_fp32 = fused_temporal_fp32 == "1"
            module._inference_native_fp32_temporal_attention = native_fp32 == "1"
    print(f"FUSED_TEMPORAL_ATTENTION_FP32 requested={fused_temporal_fp32} "
          "tf32_stage=native", flush=True)
    print(f"NATIVE_FP32_TEMPORAL_ATTENTION requested={native_fp32}",flush=True)
    fused_residual_norm = os.environ.get("UNIFOLM_FUSED_RESIDUAL_NORM", "0")
    if fused_residual_norm not in {"0", "1"}:
        raise ValueError("UNIFOLM_FUSED_RESIDUAL_NORM must be 0 or 1")
    from unifolm_wma.modules.attention import BasicTransformerBlock
    fused_residual_norm_count = 0
    for module in model.model.diffusion_model.modules():
        if isinstance(module, BasicTransformerBlock):
            module._inference_fused_residual_norm = fused_residual_norm == "1"
            fused_residual_norm_count += 1
    print(f"FUSED_RESIDUAL_NORM requested={fused_residual_norm} "
          f"modules={fused_residual_norm_count}", flush=True)
    direct_temporal_layout = os.environ.get("UNIFOLM_TEMPORAL_LAYOUT_DIRECT", "0")
    if direct_temporal_layout not in {"0", "1"}:
        raise ValueError("UNIFOLM_TEMPORAL_LAYOUT_DIRECT must be 0 or 1")
    from unifolm_wma.modules.attention import TemporalTransformer
    direct_temporal_count = 0
    for module in model.model.diffusion_model.modules():
        if isinstance(module, TemporalTransformer):
            module._inference_direct_layout = direct_temporal_layout == "1"
            direct_temporal_count += 1
    print(f"TEMPORAL_LAYOUT_DIRECT requested={direct_temporal_layout} "
          f"modules={direct_temporal_count}", flush=True)
    if split_norm not in {"0", "1", "fp32"}:
        raise ValueError("UNIFOLM_SPLIT_GROUP_NORM must be 0, 1, or fp32")
    split_norm_count = 0
    if split_norm != "0":
        if split_norm == "1" and (
                not full_fp16_enabled or full_fp16_norm_dtype != "fp16"):
            raise ValueError("Split GroupNorm requires FP16 denoiser and norms")
        if split_norm == "fp32" and full_fp16_enabled:
            raise ValueError("FP32 split GroupNorm requires FP32 denoiser")
        if fused_norm_enabled or framewise_norm_enabled or exact_temporal_norm_enabled:
            raise ValueError("Split GroupNorm cannot overlap other norm fusions")
        from unifolm_wma.utils.triton_split_group_norm import install_split_group_norm
        split_norm_count = install_split_group_norm(model.model.diffusion_model)
    print(f"SPLIT_GROUP_NORM requested={split_norm} modules={split_norm_count}",
          flush=True)
    split_transformer_norm = os.environ.get("UNIFOLM_SPLIT_TRANSFORMER_NORM", "0")
    if split_transformer_norm not in {"0", "1"}:
        raise ValueError("UNIFOLM_SPLIT_TRANSFORMER_NORM must be 0 or 1")
    split_transformer_norm_count = 0
    if split_transformer_norm == "1":
        if split_norm == "0":
            raise ValueError("Split transformer norm requires split GroupNorm")
        from unifolm_wma.utils.triton_split_group_norm import install_split_transformer_norms
        split_transformer_norm_count = install_split_transformer_norms(
            model.model.diffusion_model)
    print(f"SPLIT_TRANSFORMER_NORM requested={split_transformer_norm} "
          f"modules={split_transformer_norm_count}", flush=True)
    state_conv1d_mode = os.environ.get("UNIFOLM_SIM_HEAD_CONV1D", "native")
    if state_conv1d_mode not in {"native", "gemm"}:
        raise ValueError(
            "UNIFOLM_SIM_HEAD_CONV1D must be native or gemm")
    from unifolm_wma.models.diffusion_head.conv1d_components import Conv1dBlock
    state_conv1d_count = 0
    for module in model.model.diffusion_model.state_unet.modules():
        if type(module) is Conv1dBlock:
            module._inference_gemm = state_conv1d_mode == "gemm"
            state_conv1d_count += 1
    print(f"SIM_HEAD_CONV1D requested={state_conv1d_mode} "
          f"modules={state_conv1d_count}", flush=True)
    if fp32_video_boundaries != "0":
        from unifolm_wma.utils.mixed_precision_feedback import install_fp32_video_output
        install_fp32_video_output(model.model.diffusion_model)
        print("FP32_VIDEO_OUTPUT installed_after_operator_fusions=true", flush=True)
    if fp32_temporal_blocks != "0":
        from unifolm_wma.utils.mixed_precision_feedback import install_fp32_temporal_convs
        count = install_fp32_temporal_convs(model.model.diffusion_model)
        print(f"FP32_TEMPORAL_BLOCKS requested={fp32_temporal_blocks} modules={count} "
              "fp32_stage=conv2d tf32_stage=original_conv3d", flush=True)
    if backbone_amp != "0":
        from unifolm_wma.utils.backbone_amp import install_backbone_amp
        counts = install_backbone_amp(model.model.diffusion_model)
        print(f"BACKBONE_AMP requested={backbone_amp} protected={counts} "
              "parameters=fp32 residuals=fp32 norms=fp32 feedback=fp32 "
              "weight_cast_cache=false", flush=True)
    primary_geglu_mode=os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU','0')
    primary_geglu_scope=os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU_SCOPE','all')
    if primary_geglu_mode not in {'0','value','both'} or primary_geglu_scope not in {'all','640','1280'}:
        raise ValueError('Invalid decision primary GEGLU mode or scope')
    primary_geglu_count=0
    if primary_geglu_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                prepacked_geglu_mode!='1'):
            raise ValueError('Primary GEGLU requires accepted V2 decision-cudnn and wideGEGLU')
        from .decision_primary_geglu import install_decision_primary_geglus
        primary_geglu_count=install_decision_primary_geglus(model.model.diffusion_model,
            primary_geglu_mode,primary_geglu_scope)
    print(f'DECISION_PRIMARY_GEGLU requested={primary_geglu_mode} '
          f'channel_scope={primary_geglu_scope} modules={primary_geglu_count} '
          'world=accepted3Half master_bias_gelu_output=fp32 '
          'weights=first_timed_warm inputs=refreshed_each_timed_call',flush=True)
    pointwise_nchw_mode=os.environ.get('UNIFOLM_POINTWISE_NCHW','0')
    if pointwise_nchw_mode not in {'0','1'}:
        raise ValueError('UNIFOLM_POINTWISE_NCHW must be0 or1')
    pointwise_nchw_count=0
    if pointwise_nchw_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Pointwise NCHW requires V2 FP32 decision-cudnn stage policy')
        from .pointwise_nchw_runtime import install_pointwise_nchw
        pointwise_nchw_count=install_pointwise_nchw(model.model.diffusion_model)
    print(f'POINTWISE_NCHW requested={pointwise_nchw_mode} modules={pointwise_nchw_count} '
          'scope=world_10_captured_shapes decision_head=native BK=64 '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'input_output_layout=direct_nchw',flush=True)
    initial_world_mode=os.environ.get('UNIFOLM_INITIAL_WORLD_LINEARS','0')
    if initial_world_mode not in {'0','1'}:
        raise ValueError('Initial world linears must be0 or1')
    initial_world_count=0
    if initial_world_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                initial_decision_mode!='1'):
            raise ValueError('Initial world linears require accepted V2 decision wrappers')
        from .initial_world_linears import install_initial_world_linears
        initial_world_count=install_initial_world_linears(model.model.diffusion_model)
    print(f'INITIAL_WORLD_LINEARS requested={initial_world_mode} modules={initial_world_count} '
          'BK=32 scope=world_initial512 decision=previous_installed_path '
          'master_bias_accum_output=fp32 weights=first_timed_warm',flush=True)
    primary_ffn_mode=os.environ.get('UNIFOLM_DECISION_PRIMARY_FFN','0')
    if primary_ffn_mode not in {'0','all','640','1280'}:
        raise ValueError('UNIFOLM_DECISION_PRIMARY_FFN must be0,all,640,or1280')
    primary_ffn_count=0
    if primary_ffn_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                vendor_ffn_mode!='chunked'):
            raise ValueError('Primary FFN requires accepted V2 decision chunked FFN policy')
        from .decision_primary_ffn import install_decision_primary_ffns
        primary_ffn_count=install_decision_primary_ffns(model.model.diffusion_model,primary_ffn_mode)
    print(f'DECISION_PRIMARY_FFN requested={primary_ffn_mode} modules={primary_ffn_count} '
          'scope=decision_ffn_output world=accepted_compensation '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'input_conversion=each_timed_call',flush=True)
    world_narrow_geglu_mode=os.environ.get('UNIFOLM_WORLD_PREPACKED_NARROW_GEGLU','0')
    if world_narrow_geglu_mode not in {'0','1'}:
        raise ValueError('World narrow GEGLU must be0 or1')
    world_narrow_geglu_tile=tuple(int(v) for v in os.environ.get('UNIFOLM_WORLD_NARROW_GEGLU_TILE','64,32,16').split(','))
    world_narrow_geglu_count=0
    if world_narrow_geglu_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                decision_half_geglu_mode!='1'):
            raise ValueError('World narrow GEGLU requires accepted V2 decision stage policy')
        from .world_prepacked_narrow_geglu import install_world_prepacked_narrow_geglus
        world_narrow_geglu_count=install_world_prepacked_narrow_geglus(
            model.model.diffusion_model,world_narrow_geglu_tile)
    print(f'WORLD_PREPACKED_NARROW_GEGLU requested={world_narrow_geglu_mode} '
          f'modules={world_narrow_geglu_count} tile={world_narrow_geglu_tile} BK=64 '
          'scope=world_C320 decision=previous_installed_path '
          'master_bias_accum_gelu_output=fp32 weights=first_timed_warm '
          'input_parts=refreshed_each_timed_call',flush=True)
    packed_spatial_mode=os.environ.get('UNIFOLM_PACKED_SPATIAL_CONV','0')
    if packed_spatial_mode not in {'0','all','1280'}:
        raise ValueError('Packed spatial scope must be0,all,or1280')
    packed_spatial_count=0
    if packed_spatial_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Packed spatial convolution requires accepted V2 stage policy')
        from .packed_spatial_runtime import install_packed_spatial_convs
        packed_spatial_count=install_packed_spatial_convs(model.model.diffusion_model,packed_spatial_mode)
    print(f'PACKED_SPATIAL_CONV requested={packed_spatial_mode} modules={packed_spatial_count} '
          'scope=world5x8 decision_medium=previous_installed_path BK=64 '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'patch_parts=refreshed_each_timed_call splitK=1_or4 deterministic_reduce=true',flush=True)
    c320_primary_geglu_mode=os.environ.get('UNIFOLM_DECISION_C320_PRIMARY_GEGLU','0')
    if c320_primary_geglu_mode not in {'0','both','value','gate'}:
        raise ValueError('Decision C320 GEGLU mode must be0,both,value,orgate')
    c320_primary_geglu_bn=int(os.environ.get('UNIFOLM_DECISION_C320_GEGLU_BN','64'))
    c320_primary_geglu_count=0
    if c320_primary_geglu_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                decision_half_geglu_mode!='1' or world_narrow_geglu_mode!='1'):
            raise ValueError('C320 primary GEGLU requires accepted V2 compensated wrappers')
        from .decision_c320_primary_geglu import install_decision_c320_primary_geglus
        c320_primary_geglu_count=install_decision_c320_primary_geglus(
            model.model.diffusion_model,c320_primary_geglu_mode,c320_primary_geglu_bn)
        if c320_primary_geglu_count!=10:
            raise ValueError('Expected ten C320 GEGLU modules')
    print(f'DECISION_C320_PRIMARY_GEGLU requested={c320_primary_geglu_mode} '
          f'modules={c320_primary_geglu_count} BN={c320_primary_geglu_bn} BK=32 '
          'scope=decision_C320 world=previous_installed_path '
          'master_bias_accum_gelu_output=fp32 weights=first_timed_warm '
          'input_parts=refreshed_each_timed_call',flush=True)
    primary_narrow_mode=os.environ.get('UNIFOLM_DECISION_PRIMARY_NARROW','0')
    if primary_narrow_mode not in {'0','qkv','output','all'}:
        raise ValueError('Decision primary narrow scope must be0,qkv,output,orall')
    primary_narrow_count=0
    if primary_narrow_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Primary C320 projections require accepted V2 decision policy')
        from .decision_primary_narrow import install_decision_primary_narrow
        primary_narrow_count=install_decision_primary_narrow(model.model.diffusion_model,primary_narrow_mode)
        if primary_narrow_count==0:
            raise ValueError('No eligible C320 projections found')
    print(f'DECISION_PRIMARY_NARROW requested={primary_narrow_mode} modules={primary_narrow_count} '
          'shape=40960x320x320 world=previous_installed_path '
          'master_bias_accum_output=fp32 weights=first_timed_warm '
          'input_cast_and_epilogue=fused_each_timed_call',flush=True)
    world_c320_primary_mode=os.environ.get('UNIFOLM_WORLD_C320_BK64_GEGLU','0')
    if world_c320_primary_mode not in {'0','both','value','gate'}:
        raise ValueError('World C320 GEGLU mode must be0,both,value,orgate')
    world_c320_primary_count=0
    if world_c320_primary_mode!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                c320_primary_geglu_mode!='both' or world_narrow_geglu_mode!='1'):
            raise ValueError('World C320 experiment requires accepted V2 wrappers')
        from .world_bk64_c320_geglu import install_world_c320_primary_geglus
        world_c320_primary_count=install_world_c320_primary_geglus(
            model.model.diffusion_model,world_c320_primary_mode,32 if world_c320_primary_mode=='value' else 64)
        if world_c320_primary_count!=10:
            raise ValueError('Expected ten world C320 GEGLU modules')
    print(f'WORLD_BK64_C320_GEGLU requested={world_c320_primary_mode} '
          f'modules={world_c320_primary_count} BN=mode_selected BK=64 '
          'scope=world_C320 decision=previous_installed_path '
          'master_bias_accum_gelu_output=fp32 weights=first_timed_warm '
          'input_parts=refreshed_each_timed_call',flush=True)
    world_wide_gate_scope=os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY','0')
    if world_wide_gate_scope not in {'0','all','640','1280'}:
        raise ValueError('Invalid world wide gate scope')
    world_wide_gate_count=0
    if world_wide_gate_scope!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                prepacked_geglu_mode!='1'):
            raise ValueError('World wide gate requires accepted V2 policy')
        from .world_wide_gate_geglu import install_world_wide_gate_geglus
        world_wide_gate_count=install_world_wide_gate_geglus(model.model.diffusion_model,
            'gate',world_wide_gate_scope)
        if world_wide_gate_count!={'all':22,'640':10,'1280':12}[world_wide_gate_scope]:
            raise ValueError('Unexpected world wide gate module count')
    print(f'WORLD_WIDE_GATE_PRIMARY scope={world_wide_gate_scope} '
          f'modules={world_wide_gate_count} value_branch=accepted_3HalfBK64 '
          'gate_branch=primary master_bias_accum_gelu_output=fp32 '
          'decision=previous_installed_path weights=first_timed_warm',flush=True)
    initial_primary_scope=os.environ.get('UNIFOLM_DECISION_INITIAL_PRIMARY','0')
    if initial_primary_scope not in {'0','projections','ffn','all'}:
        raise ValueError('Invalid initial512 primary scope')
    initial_primary_count=0
    if initial_primary_scope!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Initial512 primary requires accepted V2 stage policy')
        from .decision_initial_primary import install_decision_initial_primary
        initial_primary_count=install_decision_initial_primary(model.model.diffusion_model,
            initial_primary_scope)
        if initial_primary_count!={'projections':8,'ffn':2,'all':10}[initial_primary_scope]:
            raise ValueError('Unexpected initial512 primary module count')
    print(f'DECISION_INITIAL_PRIMARY scope={initial_primary_scope} modules={initial_primary_count} '
          'world=previous master_bias_accum_output=fp32 '
          'static_weight_parts=first_timed_warm input_conversion=each_timed_call',flush=True)
    initial_primary_role=os.environ.get('UNIFOLM_DECISION_INITIAL_PRIMARY_ROLE','0')
    if initial_primary_role not in {'0','qk','vo'}:
        raise ValueError('Initial primary role must be0,qk,orvo')
    initial_primary_role_count=0
    if initial_primary_role!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or initial_primary_scope not in {'0','ffn'} or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Initial role experiment requires separate V2 stage policy')
        from .decision_initial_projection_roles import install_decision_initial_projection_roles
        initial_primary_role_count=install_decision_initial_projection_roles(
            model.model.diffusion_model,initial_primary_role)
        if initial_primary_role_count!=4:
            raise ValueError('Unexpected initial512 attention projection role count')
    print(f'DECISION_INITIAL_PRIMARY_ROLE role={initial_primary_role} '
          f'modules={initial_primary_role_count} other_roles_world=previous '
          'master_bias_accum_output=fp32 input_conversion=each_timed_call '
          'weights=first_timed_warm',flush=True)
    initial_primary_geglu_mode=os.environ.get('UNIFOLM_INITIAL_PRIMARY_GEGLU','0')
    if initial_primary_geglu_mode not in {'0','1'}:
        raise ValueError('Initial primary GEGLU must be0or1')
    initial_primary_geglu_count=0
    if initial_primary_geglu_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or initial_primary_scope!='ffn' or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_NVRTC_GEGLU','0')!='fmad'):
            raise ValueError('Initial primary fusion requires accepted FFN2 policy/native activation')
        from .initial_primary_geglu import install_initial_primary_geglu
        initial_primary_geglu_count=install_initial_primary_geglu(model.model.diffusion_model)
        if initial_primary_geglu_count!=1:
            raise ValueError('Unexpected initial512 primary GEGLU count')
    print(f'INITIAL_PRIMARY_GEGLU requested={initial_primary_geglu_mode} '
          f'modules={initial_primary_geglu_count} stage=decision tile=64,128,32,4,2 '
          'world=previous activation=native_sm80 master_bias_accum_output=fp32 '
          'weights=first_timed_warm input_cast=each_timed_call',flush=True)
    half_hidden_initial_mode=os.environ.get('UNIFOLM_HALF_HIDDEN_INITIAL_FFN','0')
    if half_hidden_initial_mode not in {'0','1'}:
        raise ValueError('Half hidden initial FFN mode must be0or1')
    half_hidden_initial_count=0
    if half_hidden_initial_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or initial_primary_scope!='ffn' or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_INITIAL_PRIMARY_GEGLU','0')!='0' or
                os.environ.get('UNIFOLM_NVRTC_GEGLU','0')!='fmad'):
            raise ValueError('Half hidden FFN requires separate accepted primaryFFN/native policy')
        from .half_hidden_initial_ffn import install_half_hidden_initial_ffn
        half_hidden_initial_count=install_half_hidden_initial_ffn(model.model.diffusion_model)
        if half_hidden_initial_count!=1:
            raise ValueError('Unexpected initial FFN count')
    print(f'HALF_HIDDEN_INITIAL_FFN requested={half_hidden_initial_mode} '
          f'modules={half_hidden_initial_count} world=previous '
          'hidden=existing_final_gemm_half_input master_bias_accum_output=fp32 '
          'weights=first_timed_warm dynamic_input=each_timed_call',flush=True)
    primary_grouped_qkv_mode=os.environ.get('UNIFOLM_DECISION_PRIMARY_GROUPED_QKV','0')
    primary_grouped_qkv_scope=os.environ.get('UNIFOLM_DECISION_PRIMARY_GROUPED_QKV_SCOPE','temporal')
    if primary_grouped_qkv_scope not in {'temporal','all'}:
        raise ValueError('Invalid primary grouped QKV scope')
    if primary_grouped_qkv_mode not in {'0','1'}:
        raise ValueError('Primary grouped QKV mode must be0or1')
    primary_grouped_qkv_count=0
    if primary_grouped_qkv_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_DECISION_PRIMARY_NARROW','0') not in {'all','qkv'}):
            raise ValueError('Primary grouped QKV requires accepted primary C320 decision projections')
        from .decision_primary_grouped_qkv import install_decision_primary_grouped_qkv
        primary_grouped_qkv_count=install_decision_primary_grouped_qkv(model.model.diffusion_model,scope=primary_grouped_qkv_scope)
        if primary_grouped_qkv_count!=(10 if primary_grouped_qkv_scope=='temporal' else 15):
            raise ValueError(f'Unexpected primary grouped QKV count: {primary_grouped_qkv_count}')
    print(f'DECISION_PRIMARY_GROUPED_QKV requested={primary_grouped_qkv_mode} '
          f'modules={primary_grouped_qkv_count} scope={primary_grouped_qkv_scope} world=previous master_accum_output=fp32 '
          'weights=first_timed_warm dynamic_input=each_timed_call',flush=True)
    two_product_protection = os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_OUTPUT', '0')
    if two_product_protection not in {'0', 'input', 'weight'}:
        raise ValueError('Invalid world two-product output protection')
    two_product_count = 0
    if two_product_protection != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0 or
                primary_narrow_mode != 'all' or
                os.environ.get('UNIFOLM_WORLD_C320_PRIMARY_LINEAR', '0') != '0'):
            raise ValueError('Two-product world outputs require accepted V2 stage policy')
        from .world_two_product_output import install_world_two_product_output
        two_product_count = install_world_two_product_output(
            model.model.diffusion_model, two_product_protection)
        if two_product_count == 0:
            raise ValueError('No eligible world two-product output projections')
    print(f'WORLD_TWO_PRODUCT_OUTPUT protect={two_product_protection} modules={two_product_count} '
          'shape=40960x320x320 bk=64 products=2 master_bias_accum_output=fp32 '
          'weights=first_timed_warm dynamic_input=each_timed_call '
          'decision_and_other_world_roles=previous_installed_path', flush=True)
    two_product_role = os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE', '0')
    two_product_role_protect = os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT', 'input')
    if (two_product_role not in {'0', 'qk320', 'output640', 'ffn640', 'output640_ffn640'} or
            two_product_role_protect not in {'input', 'weight'}):
        raise ValueError('Invalid world two-product role/protection')
    two_product_role_count = 0
    if two_product_role != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0 or
                os.environ.get('UNIFOLM_WORLD_C320_PRIMARY_LINEAR', '0') != '0'):
            raise ValueError('World two-product role requires accepted V2 stage policy')
        from .world_two_product_roles import install_world_two_product_roles
        two_product_role_scopes = (('output640', 'ffn640')
            if two_product_role == 'output640_ffn640' else (two_product_role,))
        for two_product_role_scope in two_product_role_scopes:
            two_product_role_count += install_world_two_product_roles(
                model.model.diffusion_model, two_product_role_scope, two_product_role_protect)
        if two_product_role_count == 0:
            raise ValueError('No eligible world two-product roles found')
    print(f'WORLD_TWO_PRODUCT_ROLE scope={two_product_role} protect={two_product_role_protect} '
          f'modules={two_product_role_count} bk=64 products=2 master_bias_accum_output=fp32 '
          'weights=first_timed_warm version_refresh=same_storage '
          'dynamic_input=each_timed_call decision_and_other_roles=previous', flush=True)
    world_zero_weight_low = os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW', '0')
    if world_zero_weight_low not in {'0', '1'}:
        raise ValueError('Invalid zero-weight-residual mode')
    world_zero_weight_low_count = 0
    if world_zero_weight_low == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('World zero residual requires accepted V2 stage policy')
        from .world_zero_weight_low_linear import install_world_zero_weight_low_linears
        world_zero_weight_low_count = install_world_zero_weight_low_linears(model.model.diffusion_model)
        if world_zero_weight_low_count == 0:
            raise ValueError('No eligible world zero-weight-residual linears')
    print(f'WORLD_ZERO_WEIGHT_LOW enabled={world_zero_weight_low} modules={world_zero_weight_low_count} '
          'only=world_M2560_640_N1280_K1280_5120 predicate=device_outside_K_loop '
          'rounding=previous_per_BK64 nonzero_residual=previous3products '
          'input_pack=each_timed_call weights_and_predicate=first_timed_warm '
          'version_refresh=same_storage master_bias_accum_output=fp32', flush=True)
    world_zero_qkv_scope = os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV', '0')
    if world_zero_qkv_scope not in {'0', 'all', '320', 'wide'}:
        raise ValueError('Invalid world zero-weight-residual QKV scope')
    world_zero_qkv_count = 0
    if world_zero_qkv_scope != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('World QKV residual optimization requires accepted V2 stage policy')
        from .world_zero_weight_low_qkv import install_world_zero_weight_low_qkv
        world_zero_qkv_count = install_world_zero_weight_low_qkv(model.model.diffusion_model, world_zero_qkv_scope)
        if world_zero_qkv_count == 0:
            raise ValueError('No eligible world QKV modules')
    print(f'WORLD_ZERO_WEIGHT_LOW_QKV scope={world_zero_qkv_scope} modules={world_zero_qkv_count} '
          'rounding=previous_per_BK64 predicate=device_outside_K_loop '
          'nonzero_residual=previous3products input_pack=each_timed_call '
          'weights_and_predicate=first_timed_warm version_refresh=same_storage '
          'master_accum_and_3_contiguous_output_planes=fp32 decision=previous', flush=True)
    concat_f2_scope = os.environ.get('UNIFOLM_WORLD_CONCAT_F2', '0')
    if concat_f2_scope not in {'0', 'all', '960', '1280', '1920'}:
        raise ValueError('Invalid world concat F2 scope')
    concat_f2_count = 0
    if concat_f2_scope != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0):
            raise ValueError('Concat F2 requires accepted V2 stage policy')
        from .world_concat_f2 import install_world_concat_f2
        concat_f2_count = install_world_concat_f2(model.model.diffusion_model, concat_f2_scope)
        if concat_f2_count == 0:
            raise ValueError('No eligible world concat F2 convolutions')
    print(f'WORLD_CONCAT_F2 scope={concat_f2_scope} modules={concat_f2_count} '
          'only=world_N16_CO640_CI960_1280_1920_H20_W32 '
          'products=3 transforms_accum_master_bias_output=fp32 block_k=64 '
          'input_transform=each_timed_call filter=first_timed_warm '
          'version_refresh=same_storage all_other_paths=previous', flush=True)
    temporal_zero_scope = os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_TEMPORAL', '0')
    if temporal_zero_scope not in {'0', 'all', 'implicit', 'packed'}:
        raise ValueError('Invalid temporal residual scope')
    temporal_zero_count = 0
    if temporal_zero_scope != '0':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0 or
                os.environ.get('UNIFOLM_COMPENSATED_TEMPORAL_CONV') != '1' or
                os.environ.get('UNIFOLM_IMPLICIT_TEMPORAL_CONV') != '1'):
            raise ValueError('Temporal residual optimization requires accepted V2 stage/conv policy')
        from .world_zero_weight_low_temporal import install_world_zero_weight_low_temporal
        temporal_zero_count = install_world_zero_weight_low_temporal(model.model.diffusion_model, temporal_zero_scope)
        if temporal_zero_count == 0:
            raise ValueError('No eligible accepted temporal conv wrappers')
    print(f'WORLD_ZERO_WEIGHT_LOW_TEMPORAL scope={temporal_zero_scope} modules={temporal_zero_count} '
          'rounding=previous_per_BK64 predicate=device_outside_K_loop nonzero_residual=previous3products '
          'C320=implicit_g8 C640=implicit_g16 C1280=packed '
          'input_conversion=each_timed_call weights_and_predicate=first_timed_warm '
          'version_refresh=same_storage master_bias_accum_output=fp32 decision=previous', flush=True)
    initial_qkv_mode = os.environ.get('UNIFOLM_INITIAL_BK32_QKV', '0')
    if initial_qkv_mode not in {'0', '1'}:
        raise ValueError('Invalid initial512 BK32 QKV mode')
    initial_qkv_count = 0
    if initial_qkv_mode == '1':
        if (dataset != STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode() != 'decision-cudnn' or stage_tf32_ddim_prefix_steps() != 0 or
                os.environ.get('UNIFOLM_INITIAL_WORLD_LINEARS') != '1'):
            raise ValueError('Initial512 QKV requires accepted V2 stage policy')
        from .initial_bk32_qkv import install_initial_bk32_qkv
        initial_qkv_count = install_initial_bk32_qkv(model.model.diffusion_model)
        if initial_qkv_count != 2:
            raise ValueError('Expected exactly two initial512 self-attention modules')
    print(f'INITIAL_BK32_QKV enabled={initial_qkv_mode} modules={initial_qkv_count} '
          'only=world_initial512_M40960 rounding=previous_per_BK32 '
          'input_pack=once_per_QKV_call static_weights_and_predicate=first_timed_warm '
          'nonzero_residual=previous3products version_refresh=same_storage '
          'master_accum_output=fp32 layout=3_contiguous_planes decision=previous', flush=True)
    initial_native_mode=os.environ.get('UNIFOLM_INITIAL_BK32_NATIVE_GEGLU','0')
    if initial_native_mode not in {'0','1'}:
        raise ValueError('Invalid initial BK32 native GEGLU mode')
    initial_native_count=0
    if initial_native_mode=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or _stage_tf32_mode()!='decision-cudnn'
                or stage_tf32_ddim_prefix_steps()!=0 or os.environ.get('UNIFOLM_INITIAL_WORLD_LINEARS')!='1'):
            raise ValueError('Initial BK32 native GEGLU requires accepted V2 stage policy')
        from .initial_bk32_native_geglu import install_initial_bk32_native_geglu
        initial_native_count=install_initial_bk32_native_geglu(model.model.diffusion_model)
        if initial_native_count==0:raise ValueError('No initial512 world FFN selected')
    print(f'INITIAL_BK32_NATIVE_GEGLU enabled={initial_native_mode} modules={initial_native_count} '
          'scope=world_initial512_M40960 original_BK32_three_products=true native_activation=true '
          'weight_pack=first_timed_warm version_refresh=same_storage input_conversion=each_call '
          'master_bias_accum_output=fp32 decision=previous',flush=True)
    qkv_attention_scope=os.environ.get('UNIFOLM_QKV_PACKED_ATTENTION','0')
    if qkv_attention_scope not in {'0','all','c320','c640'}:
        raise ValueError('Invalid QKV packed attention scope')
    qkv_attention_count=0
    if qkv_attention_scope!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or
                os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV')!='all' or
                os.environ.get('UNIFOLM_COMPENSATED_SPATIAL_ATTENTION')!='large' or
                os.environ.get('UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV')!='kv'):
            raise ValueError('QKV attention fusion requires accepted V2 world QKV/attention policy')
        from .world_qkv_packed_attention import install_world_qkv_packed_attention
        qkv_attention_count=install_world_qkv_packed_attention(model.model.diffusion_model,qkv_attention_scope)
        if qkv_attention_count!=(10 if qkv_attention_scope=='all' else 5):
            raise ValueError('Unexpected spatial QKV attention module count')
    print(f'QKV_PACKED_ATTENTION_MODEL scope={qkv_attention_scope} modules={qkv_attention_count} '
          'QKV=previous_BK64_group32_device_residual_predicate '
          'K_V_split=fused_previous_math attention=previous_BM64_BN64 '
          'master_public_output=fp32 input_pack=each_call static_weights=first_timed_warm '
          'cache=previous_shared_QKV_same_storage decision=previous',flush=True)
    decision_qkv_attention=os.environ.get('UNIFOLM_DECISION_QKV_PACKED_ATTENTION','0')
    if decision_qkv_attention not in {'0','1'}:
        raise ValueError('Invalid decision QKV packed attention mode')
    decision_qkv_attention_count=0
    if decision_qkv_attention=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_DECISION_PRIMARY_GROUPED_QKV')!='1' or
                os.environ.get('UNIFOLM_DECISION_PRIMARY_GROUPED_QKV_SCOPE')!='all' or
                os.environ.get('UNIFOLM_QKV_PACKED_ATTENTION')!='all' or
                os.environ.get('UNIFOLM_COMPENSATED_SPATIAL_ATTENTION')!='large' or
                os.environ.get('UNIFOLM_SPATIAL_ATTENTION_PREPACK_KV')!='kv'):
            raise ValueError('Decision QKV fusion requires accepted V2 primary QKV and attention')
        from .decision_qkv_packed_attention import install_decision_qkv_packed_attention
        decision_qkv_attention_count=install_decision_qkv_packed_attention(model.model.diffusion_model)
        if decision_qkv_attention_count!=5:
            raise ValueError('Unexpected decision C320 spatial QKV count')
    print(f'DECISION_QKV_PACKED_ATTENTION_MODEL enabled={decision_qkv_attention} '
          f'modules={decision_qkv_attention_count} primary_BM128_BN128_BK64_group32_stages3=true '
          'attention=previous_BM64_BN64 master_public_output=fp32 input_conversion=each_call '
          'static_weights=first_timed_warm cache=previous_primary_QKV_same_storage world=previous',flush=True)
    norm3_pack_scope=os.environ.get('UNIFOLM_WORLD_NORM3_PACKED_GEGLU','0')
    if norm3_pack_scope not in {'0','all','c320','c640','c1280'}:
        raise ValueError('Invalid world norm3 GEGLU packing scope')
    norm3_pack_count=0
    if norm3_pack_scope!='0':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_FUSED_RESIDUAL_NORM')!='1' or
                os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU')!='all' or
                os.environ.get('UNIFOLM_WORLD_C320_BK64_GEGLU')!='gate' or
                os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY')!='640' or
                os.environ.get('UNIFOLM_PREPACKED_GEGLU')!='1'):
            raise ValueError('Norm3 packing requires accepted residual norm and world GEGLU policy')
        from .world_norm3_packed_geglu import install_world_norm3_packed_geglu
        norm3_pack_count=install_world_norm3_packed_geglu(model.model.diffusion_model,norm3_pack_scope)
        if norm3_pack_count!={'all':32,'c320':10,'c640':10,'c1280':12}[norm3_pack_scope]:
            raise ValueError('Unexpected norm3 GEGLU block count')
    print(f'WORLD_NORM3_PACKED_GEGLU_MODEL scope={norm3_pack_scope} blocks={norm3_pack_count} '
          'norm_math=previous_reduction_layout4_half_store_address_identity '
          'GEGLU=previous_JIT_BK64_BM_BN_group_primary_modes master_public_output=fp32 '
          'dynamic_norm_packing=each_call weight_pack=first_timed_warm_version_refresh_same_storage '
          'decision=previous initial512=previous',flush=True)
    ffn_residual_mode=os.environ.get('UNIFOLM_WORLD_FFN_RESIDUAL_EPILOGUE','0')
    if ffn_residual_mode not in {'0','1'}:raise ValueError('Invalid FFN residual epilogue mode')
    ffn_residual_count=0
    if ffn_residual_mode=='1':
        required={'UNIFOLM_WORLD_NORM3_PACKED_GEGLU':'all',
            'UNIFOLM_COMPENSATED_LINEAR_BACKEND':'scaled-fp16',
            'UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE':'1',
            'UNIFOLM_COMPENSATED_LINEAR_GROUP_M':'32',
            'UNIFOLM_COMPENSATED_LINEAR_BLOCK_K':'64',
            'UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK':'wide',
            'UNIFOLM_TF32X3_NARROW_OUTPUT':'1'}
        if any(os.environ.get(k)!=v for k,v in required.items()):
            raise ValueError('FFN residual epilogue requires verified world norm3 and original C320 linear policy')
        from .world_ffn_residual_epilogue import install_world_ffn_residual_epilogue
        ffn_residual_count=install_world_ffn_residual_epilogue(model.model.diffusion_model)
        if ffn_residual_count!=10:raise ValueError('Unexpected C320 FFN residual epilogue count')
    print(f'WORLD_FFN_RESIDUAL_EPILOGUE_MODEL mode={ffn_residual_mode} blocks={ffn_residual_count} '
          'GEMM=previous_three_products_BK64_BM64_BN64_group32_stages2 '
          'residual_add=fp32_before_store weight_pack=first_timed_warm_version_refresh_same_storage '
          'decision=previous masters=fp32',flush=True)
    ffn640_residual_mode=os.environ.get('UNIFOLM_WORLD_FFN640_RESIDUAL_EPILOGUE','0')
    if ffn640_residual_mode not in {'0','1'}:raise ValueError('Invalid C640 FFN residual mode')
    ffn640_residual_count=0
    if ffn640_residual_mode=='1':
        if (norm3_pack_scope!='all' or ffn_residual_mode!='1' or
                os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE')!='output640_ffn640' or
                os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT')!='input'):
            raise ValueError('C640 residual fusion requires verified norm3/C320 and input-protected C640 FFN')
        from .world_ffn640_residual_epilogue import install_world_ffn640_residual_epilogue
        ffn640_residual_count=install_world_ffn640_residual_epilogue(model.model.diffusion_model)
        if ffn640_residual_count!=10:raise ValueError('Unexpected C640 FFN residual count')
    print(f'WORLD_FFN640_RESIDUAL_EPILOGUE_MODEL mode={ffn640_residual_mode} blocks={ffn640_residual_count} '
          'GEMM=previous_two_input_protected_products_BK64_BM64_BN64_group32_stages2 '
          'residual=fp32_before_store static_pack=first_timed_warm_version_refresh_same_storage',flush=True)
    decision_ffn_residual=os.environ.get('UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE','0')
    if decision_ffn_residual not in {'0','1'}:raise ValueError('Invalid decision FFN residual mode')
    decision_ffn_residual_count=0
    if decision_ffn_residual=='1':
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0 or
                os.environ.get('UNIFOLM_FUSED_RESIDUAL_NORM')!='1' or
                os.environ.get('UNIFOLM_FFN_DTYPE')!='fp32' or
                os.environ.get('UNIFOLM_DECISION_PRIMARY_FFN')!='1280' or
                os.environ.get('UNIFOLM_DECISION_VENDOR_FFN')!='chunked'):
            raise ValueError('Decision FFN residual fusion requires accepted primary C1280 and FP32 norm policy')
        from .decision_ffn_residual_epilogue import install_decision_ffn_residual_epilogue
        decision_ffn_residual_count=install_decision_ffn_residual_epilogue(model.model.diffusion_model)
        if decision_ffn_residual_count!=12:raise ValueError('Unexpected decision C1280 residual block count')
    print(f'DECISION_FFN_RESIDUAL_EPILOGUE_MODEL mode={decision_ffn_residual} blocks={decision_ffn_residual_count} '
          'GEMM=previous_cublas_primary scale_bias_residual=fp32 '
          'static_pack=first_timed_warm_version_refresh_same_storage world=previous',flush=True)
    world_hidden_parts=os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS','0')
    if world_hidden_parts not in {'0','1'}:raise ValueError('Invalid GEGLU hidden parts mode')
    world_hidden_parts_count=0
    if world_hidden_parts=='1':
        if (norm3_pack_scope!='all' or ffn640_residual_mode!='1' or
                os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY')!='640' or
                os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE')!='output640_ffn640' or
                os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT')!='input'):
            raise ValueError('GEGLU hidden parts requires verified C640 primary gate and protected FFN')
        from .world_geglu_hidden_parts import install_world_geglu_hidden_parts
        world_hidden_parts_count=install_world_geglu_hidden_parts(model.model.diffusion_model)
        if world_hidden_parts_count!=10:raise ValueError('Unexpected C640 hidden parts count')
    print(f'WORLD_GEGLU_HIDDEN_PARTS_MODEL mode={world_hidden_parts} blocks={world_hidden_parts_count} '
          'producer=previous_GEGLU_FP32_math_vector_Half_store consumer=previous_two_protected_products_BN128 '
          'dynamic_hidden_parts=fresh_each_call static_weights=previous_first_timed_warm_version_refresh '
          'master_public_output=fp32 decision=previous',flush=True)
    world_hidden_parts_c320=os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS_C320','0')
    if world_hidden_parts_c320 not in {'0','1'}:raise ValueError('Invalid C320 hidden parts mode')
    world_hidden_parts_c320_count=0
    if world_hidden_parts_c320=='1':
        if (world_hidden_parts!='1' or norm3_pack_scope!='all' or ffn_residual_mode!='1' or
                os.environ.get('UNIFOLM_WORLD_C320_BK64_GEGLU')!='gate'):
            raise ValueError('C320 hidden parts requires original three-product FFN and C640 pipeline')
        from .world_geglu_hidden_parts_c320 import install_world_geglu_hidden_parts_c320
        world_hidden_parts_c320_count=install_world_geglu_hidden_parts_c320(model.model.diffusion_model)
        if world_hidden_parts_c320_count!=10:raise ValueError('Unexpected C320 hidden parts count')
    print(f'WORLD_GEGLU_HIDDEN_PARTS_C320_MODEL mode={world_hidden_parts_c320} blocks={world_hidden_parts_c320_count} '
          'producer=previous_GEGLU_group16_vector_Half_store consumer=previous_three_products_BK64_BM64_BN64_group32 '
          'dynamic_hidden_parts=fresh_each_call static_weights=previous_first_timed_warm_version_refresh '
          'master_public_output=fp32 decision=previous',flush=True)
    initial_hidden_parts=os.environ.get('UNIFOLM_WORLD_INITIAL_NATIVE_HIDDEN_PARTS','0')
    if initial_hidden_parts not in {'0','1'}:raise ValueError('Invalid world initial native hidden mode')
    initial_hidden_parts_count=0
    if initial_hidden_parts=='1':
        if (os.environ.get('UNIFOLM_INITIAL_BK32_NATIVE_GEGLU')!='1'
                or os.environ.get('UNIFOLM_HALF_HIDDEN_INITIAL_FFN')!='1'
                or os.environ.get('UNIFOLM_INITIAL_WORLD_LINEARS')!='1'
                or os.environ.get('UNIFOLM_DECISION_INITIAL_PRIMARY')!='ffn'):
            raise ValueError('World initial native hidden parts requires accepted initial stage wrappers')
        from .world_initial_native_hidden_parts import install_world_initial_native_hidden_parts
        initial_hidden_parts_count=install_world_initial_native_hidden_parts(model.model.diffusion_model)
        if initial_hidden_parts_count!=1:raise ValueError('Unexpected world initial FFN count')
    print(f'WORLD_INITIAL_NATIVE_HIDDEN_PARTS_MODEL mode={initial_hidden_parts} modules={initial_hidden_parts_count} '
          'original_BK32_native_GEGLU=true producer=64x64_w4_s2 final=64x128_w4_s3_three_products '
          'dynamic_hidden_parts=fresh_each_call static_weights=first_timed_warm_version_refresh_same_storage '
          'master_public_output=fp32 decision=previous',flush=True)
    attention_hidden_scope=os.environ.get('UNIFOLM_WORLD_ATTENTION_HIDDEN_PROJECTION','0')
    if attention_hidden_scope not in {'0','all','c320','c640'}:raise ValueError('Invalid world attention hidden scope')
    attention_hidden_count=0
    if attention_hidden_scope!='0':
        if (os.environ.get('UNIFOLM_QKV_PACKED_ATTENTION')!='all'
                or os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_OUTPUT')!='input'
                or 'output640' not in os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE','').split('_ffn')[0]
                or os.environ.get('UNIFOLM_WORLD_TWO_PRODUCT_ROLE_PROTECT')!='input'):
            raise ValueError('Attention hidden projection requires original QKV/attention and input-protected output policies')
        from .world_attention_hidden_projection import install_world_attention_hidden_projection
        attention_hidden_count=install_world_attention_hidden_projection(model.model.diffusion_model,attention_hidden_scope)
        if attention_hidden_count!=(10 if attention_hidden_scope=='all' else 5):raise ValueError('Unexpected attention hidden projection count')
    print(f'WORLD_ATTENTION_HIDDEN_PROJECTION_MODEL mode={attention_hidden_scope} modules={attention_hidden_count} '
          'QKV=previous_shared_cache attention=previous_BM64_BN64_FP32_online_softmax_vector_Half_output '
          'projection=previous_input_protected_two_products_BK64_BM64_BN64_group32 '
          'dynamic_parts=fresh_each_call static_output_weights=first_timed_warm_version_refresh_same_storage '
          'master_public_output=fp32 decision=previous',flush=True)
    decision_chunked_hidden=os.environ.get('UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS','0')
    if decision_chunked_hidden not in {'0','1'}:raise ValueError('Invalid decision chunked hidden mode')
    decision_chunked_hidden_count=0
    if decision_chunked_hidden=='1':
        if (os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU')!='both'
                or os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU_SCOPE') not in {'all','640'}
                or os.environ.get('UNIFOLM_DECISION_VENDOR_FFN')!='chunked'
                or os.environ.get('UNIFOLM_DECISION_PRIMARY_FFN')!='1280'
                or os.environ.get('UNIFOLM_WORLD_NORM3_PACKED_GEGLU')!='all'
                or os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS')!='1'
                or os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY')!='640'):
            raise ValueError('Decision chunked hidden requires original C640 primary GEGLU, chunk512 FFN and world shared caches')
        from .decision_chunked_hidden_parts import install_decision_chunked_hidden_parts
        decision_chunked_hidden_count=install_decision_chunked_hidden_parts(model.model.diffusion_model)
        if decision_chunked_hidden_count!=10:raise ValueError('Unexpected decision chunked hidden count')
    print(f'DECISION_CHUNKED_HIDDEN_PARTS_MODEL mode={decision_chunked_hidden} blocks={decision_chunked_hidden_count} '
          'norm=previous_FP32_residual_norm GEGLU=previous_both_primary_BK64_BM64_BN64_group32 '
          'hidden=vector_Half_high_low GEMM=previous_chunk512_three_batched_products '
          'FP32_ordered_reduce_bias_residual=true static_weights=shared_world_GE_and_final_caches '
          'weight_pack=first_timed_warm_version_refresh_same_storage dynamic_parts=fresh_each_call '
          'master_public_output=fp32 world=previous',flush=True)
    decision_hidden_parts=os.environ.get('UNIFOLM_DECISION_PRIMARY_HIDDEN_PARTS','0')
    if decision_hidden_parts not in {'0','1'}:raise ValueError('Invalid decision primary hidden mode')
    decision_hidden_parts_count=0
    if decision_hidden_parts=='1':
        if (os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU')!='both'
                or os.environ.get('UNIFOLM_DECISION_PRIMARY_GEGLU_SCOPE') not in {'all','1280'}
                or os.environ.get('UNIFOLM_DECISION_PRIMARY_FFN') not in {'all','1280'}
                or os.environ.get('UNIFOLM_DECISION_FFN_RESIDUAL_EPILOGUE')!='1'
                or os.environ.get('UNIFOLM_WORLD_NORM3_PACKED_GEGLU')!='all'
                or os.environ.get('UNIFOLM_WORLD_WIDE_GATE_PRIMARY')!='640'):
            raise ValueError('Decision primary hidden requires original primary arithmetic and shared world GEGLU cache')
        from .decision_primary_hidden_parts import install_decision_primary_hidden_parts
        decision_hidden_parts_count=install_decision_primary_hidden_parts(model.model.diffusion_model)
        if decision_hidden_parts_count!=12:raise ValueError('Unexpected decision primary hidden count')
    print(f'DECISION_PRIMARY_HIDDEN_PARTS_MODEL mode={decision_hidden_parts} blocks={decision_hidden_parts_count} '
          'norm=previous_FP32_residual_norm GEGLU=previous_both_primary_BK64_BM64_BN32_group32 '
          'hidden=vector_Half_high GEMM=previous_cublas_primary FP32_finish_bias_residual=previous '
          'GEGLU_weight_parts=shared_world_cache final_weight_parts=shared_decision_cache '
          'weight_pack=first_timed_warm_version_refresh_same_storage dynamic_parts=fresh_each_call '
          'master_public_output=fp32 world=previous',flush=True)
    upsample_heuristic = os.environ.get('UNIFOLM_WORLD_UPSAMPLE_HEURISTIC','0')
    if upsample_heuristic not in {'0','1'}:
        raise ValueError('UNIFOLM_WORLD_UPSAMPLE_HEURISTIC must be0or1')
    upsample_heuristic_names = []
    if upsample_heuristic == '1':
        from .world_upsample_heuristic import install_world_upsample_heuristic
        upsample_heuristic_names = install_world_upsample_heuristic(model.model.diffusion_model)
    print(f'WORLD_UPSAMPLE_HEURISTIC_MODEL mode={upsample_heuristic} layers={len(upsample_heuristic_names)} '
          'scope=output_blocks.5.3.conv,output_blocks.8.3.conv world_only=true cudnn_benchmark=False '
          'precision=previous_FP32 benchmark_flag=finally_restored decision=previous',flush=True)
    world_ffn1280_zero=os.environ.get('UNIFOLM_WORLD_FFN1280_ZERO_RESIDUAL','0')
    if world_ffn1280_zero not in {'0','1'}:raise ValueError('Invalid world FFN1280 zero-residual mode')
    world_ffn1280_zero_count=0
    if world_ffn1280_zero=='1':
        required={'UNIFOLM_COMPENSATED_LINEAR_BACKEND':'scaled-fp16',
            'UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE':'1','UNIFOLM_COMPENSATED_LINEAR_GROUP_M':'32',
            'UNIFOLM_COMPENSATED_LINEAR_BLOCK_K':'64','UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK':'wide',
            'UNIFOLM_WORLD_ZERO_WEIGHT_LOW':'0','UNIFOLM_WORLD_WIDE_GATE_PRIMARY':'640',
            'UNIFOLM_WORLD_NORM3_PACKED_GEGLU':'all','UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU':'all',
            'UNIFOLM_DECISION_PRIMARY_HIDDEN_PARTS':'1','UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS':'1'}
        if any(os.environ.get(k)!=v for k,v in required.items()):
            raise ValueError('World FFN1280 zero-residual requires accepted C1280 arithmetic/shared decision cache')
        from .world_geglu_hidden_parts_c1280_zero import install_world_geglu_hidden_parts_c1280_zero
        world_ffn1280_zero_count=install_world_geglu_hidden_parts_c1280_zero(model.model.diffusion_model)
        if world_ffn1280_zero_count!=12:raise ValueError('Unexpected FFN1280 zero-residual count')
    print(f'WORLD_FFN1280_ZERO_RESIDUAL_MODEL mode={world_ffn1280_zero} blocks={world_ffn1280_zero_count} '
          'producer=previous_both_compensated_GEGLU_fresh_Half_parts '
          'consumer=previous_short_BK64_device_predicated_two_or_three_products_FP32_residual '
          'static_weights=shared_GE_and_decision_final_cache count_refresh=original_master_version_either_stage '
          'count_storage=shared_first_timed_warm_version_refresh_same_pointer '
          'cold_stale_capture=native_fallback FP32_master_public_output=true decision_math=previous',flush=True)
    world_ffn320_zero=os.environ.get('UNIFOLM_WORLD_FFN320_ZERO_RESIDUAL','0')
    if world_ffn320_zero not in {'0','1'}:raise ValueError('Invalid world FFN320 zero-residual mode')
    world_ffn320_zero_count=0
    if world_ffn320_zero=='1':
        required={'UNIFOLM_COMPENSATED_LINEAR_BACKEND':'scaled-fp16',
            'UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE':'1','UNIFOLM_COMPENSATED_LINEAR_GROUP_M':'32',
            'UNIFOLM_COMPENSATED_LINEAR_BLOCK_K':'64','UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK':'wide',
            'UNIFOLM_WORLD_ZERO_WEIGHT_LOW':'0','UNIFOLM_WORLD_NORM3_PACKED_GEGLU':'all',
            'UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU':'all','UNIFOLM_WORLD_FFN1280_ZERO_RESIDUAL':'1',
            'UNIFOLM_DECISION_PRIMARY_HIDDEN_PARTS':'1','UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS':'1'}
        if any(os.environ.get(k)!=v for k,v in required.items()):
            raise ValueError('World FFN320 zero-residual requires accepted arithmetic/cache configuration')
        from .world_ffn320_zero import install_world_ffn320_zero
        world_ffn320_zero_count=install_world_ffn320_zero(model.model.diffusion_model)
        if world_ffn320_zero_count!=10:raise ValueError('Unexpected FFN320 zero-residual count')
    print(f'WORLD_FFN320_ZERO_RESIDUAL_MODEL mode={world_ffn320_zero} blocks={world_ffn320_zero_count} '
          'producer=previous_C320_GEGLU_fresh_Half_parts '
          'consumer=accepted_short_BK64_device_predicated_two_or_three_products_FP32_residual '
          'static_weights=original_world_final_cache count_refresh=original_master_version '
          'count_storage=first_timed_world_warm_version_refresh_same_pointer '
          'cold_stale_capture=native_fallback FP32_master_public_output=true decision_math=previous',flush=True)
    geglu1280_stages=os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_STAGES','2')
    if geglu1280_stages not in {'2','3'}:raise ValueError('Invalid world C1280 GEGLU pipeline depth')
    geglu1280_count=sum(bool(getattr(m,'_world_geglu_hidden_parts_c1280_zero_installed',False))
        for m in model.model.diffusion_model.modules())
    if geglu1280_stages=='3' and (geglu1280_count!=12 or
            os.environ.get('UNIFOLM_WORLD_FFN1280_ZERO_RESIDUAL')!='1' or
            os.environ.get('UNIFOLM_WORLD_NORM3_PACKED_GEGLU')!='all' or
            os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU')!='all'):
        raise ValueError('C1280 GEGLU stage3 requires the accepted twelve world blocks')
    print(f'WORLD_GEGLU_C1280_STAGES_MODEL stages={geglu1280_stages} blocks={geglu1280_count} '
          'math=previous_BK64_BM64_BN32_group32_both_compensated '
          'shader=previous_unchanged hidden=fresh_Half_high_low '
          'weights=previous_shared_first_timed_warm_version_refresh '
          'FP32_master_bias_GELU=true decision=previous C320_C640=previous',flush=True)
    zero_qkv_stages=os.environ.get('UNIFOLM_WORLD_ZERO_QKV_STAGES','2')
    if zero_qkv_stages not in {'2','3-large'}:raise ValueError('Invalid world zero-low QKV schedule')
    zero_qkv_modules=sum(bool(getattr(m,'_world_zero_weight_low_qkv_installed',False))
        for m in model.model.diffusion_model.modules())
    if zero_qkv_stages=='3-large' and (zero_qkv_modules!=48 or
            os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV')!='all' or
            os.environ.get('UNIFOLM_COMPENSATED_LINEAR_BACKEND')!='scaled-fp16'):
        raise ValueError('Zero-low QKV schedule requires the accepted 48 world QKV modules')
    print(f'WORLD_ZERO_QKV_STAGES_MODEL mode={zero_qkv_stages} installed={zero_qkv_modules} '
          'eligible_world_calls=35 retained_small_world_calls=3 '
          'scope=three_observed_large_shapes_groups3_no_bias_world_only '
          'math=previous_short_BK64_BM64or32_BN64_group32_device_two_or_three_products '
          'weights_predicate=previous_first_timed_warm_version_refresh_same_storage '
          'input_pack=each_call output_planes=previous_FP32 decision=previous',flush=True)
    ffn1280_stages=os.environ.get('UNIFOLM_WORLD_FFN1280_ZERO_STAGES','2')
    if ffn1280_stages not in {'2','3'}:raise ValueError('Invalid world C1280 FFN depth')
    ffn1280_count=sum(bool(getattr(m,'_world_geglu_hidden_parts_c1280_zero_installed',False))
        for m in model.model.diffusion_model.modules())
    if ffn1280_stages=='3' and (ffn1280_count!=12 or
            os.environ.get('UNIFOLM_WORLD_FFN1280_ZERO_RESIDUAL')!='1'):
        raise ValueError('C1280 FFN stage3 requires the accepted twelve world blocks')
    print(f'WORLD_FFN1280_ZERO_STAGES_MODEL stages={ffn1280_stages} blocks={ffn1280_count} '
          'scope=actual_M2560_640_N1280_K5120_world_only other_shapes=depth2 '
          'math=previous_short_BK64_BM64or32_BN64_group32_two_or_three_products '
          'weights=previous_shared_first_timed_warm_version_refresh_same_count '
          'FP32_bias_residual=true decision=previous C320=previous',flush=True)
    geglu640_group=os.environ.get('UNIFOLM_WORLD_GEGLU_C640_GROUP','32')
    if geglu640_group not in {'32','4'}:raise ValueError('Invalid world C640 GEGLU group')
    geglu640_count=sum(bool(getattr(m,'_world_geglu_hidden_parts_installed',False))
        for m in model.model.diffusion_model.modules())
    if geglu640_group=='4' and (geglu640_count!=10 or
            os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS')!='1' or
            os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU')!='all'):
        raise ValueError('C640 group4 requires the accepted ten world GEGLU blocks')
    print(f'WORLD_GEGLU_C640_GROUP_MODEL group={geglu640_group} blocks={geglu640_count} '
          'scope=actual_M10240_K640_C2560_world_only other_shapes=group32 '
          'math=previous_BM64_BN64_BK64_warps4_stages2 '
          'FP32_bias_GELU=true hidden=fresh_Half_high_low '
          'weights=previous_first_timed_warm_version_refresh_same_storage decision=previous',flush=True)
    geglu_subtile32=os.environ.get('UNIFOLM_WORLD_GEGLU_SUBTILE32','0')
    if geglu_subtile32 not in {'0','1'}:raise ValueError('Invalid world GEGLU K32 mode')
    subtile320_count=sum(bool(getattr(m,'_world_ffn320_zero_installed',False))
        for m in model.model.diffusion_model.modules())
    subtile640_count=sum(bool(getattr(m,'_world_geglu_hidden_parts_installed',False))
        for m in model.model.diffusion_model.modules())
    if geglu_subtile32=='1' and (subtile320_count!=10 or subtile640_count!=10 or
            os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS')!='1' or
            os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS_C320')!='1' or
            os.environ.get('UNIFOLM_WORLD_FFN320_ZERO_RESIDUAL')!='1' or
            os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU')!='all'):
        raise ValueError('World GEGLU K32 requires the accepted twenty C320/C640 blocks')
    print(f'WORLD_GEGLU_SUBTILE32_MODEL mode={geglu_subtile32} blocks={subtile320_count+subtile640_count} '
          f'C320={subtile320_count} C640={subtile640_count} '
          'logical_BK=64 physical_BK=32 scope=two_actual_world_shapes '
          'FP32_sum_rounding_bias_GELU=previous hidden=fresh_Half_high_low '
          'device_predicate=previous_two_or_three_products '
          'weights=previous_first_timed_warm_version_refresh_same_storage '
          'other_shapes_backend_and_decision=previous C1280=previous',flush=True)
    partitioned_temporal_mode = os.environ.get("UNIFOLM_WORLD_PARTITIONED_TEMPORAL", "0")
    if partitioned_temporal_mode not in {"0", "1"}:
        raise ValueError("UNIFOLM_WORLD_PARTITIONED_TEMPORAL must be 0 or 1")
    if partitioned_temporal_mode == "1":
        from .world_partitioned_temporal import install_world_partitioned_temporal
        partitioned_temporal_count = install_world_partitioned_temporal(model.model.diffusion_model)
        if partitioned_temporal_count != 40:
            raise RuntimeError("Expected 40 narrow temporal convolution modules")
        print("WORLD_PARTITIONED_TEMPORAL_MODEL mode=1 blocks=40 C320=20 C640=20 "
              "scope=world_narrow_implicit device_predicated_launches=2 logical_BK=64 "
              "FP32_sum_rounding_bias=previous weights=first_timed_warm_version_refresh_same_storage "
              "C1280_and_decision=previous", flush=True)
    partitioned_temporal_schedule = os.environ.get("UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE", "2")
    if partitioned_temporal_schedule not in {"2", "3"}:
        raise ValueError("Invalid partitioned temporal schedule")
    if partitioned_temporal_schedule == "3":
        if os.environ.get("UNIFOLM_WORLD_PARTITIONED_TEMPORAL", "0") != "1":
            raise ValueError("Temporal scheduling requires partitioned temporal kernels")
        partitioned_schedule_modules = [m for m in model.model.diffusion_model.modules()
            if getattr(m, "_world_partitioned_temporal_installed", False)]
        if (len(partitioned_schedule_modules) != 40
                or sum(m.in_channels == 320 for m in partitioned_schedule_modules) != 20
                or sum(m.in_channels == 640 for m in partitioned_schedule_modules) != 20):
            raise RuntimeError("Expected 20 C320 and 20 C640 partitioned temporal modules")
        print("WORLD_PARTITIONED_TEMPORAL_SCHEDULE_MODEL mode=3 blocks=40 C320_group=32 "
              "C640_group=16 two_product_stages=3 three_product_stages=2 "
              "scope=actual_world_narrow_shapes general_path_and_backend_fallback=previous "
              "JIT_math_and_cache=previous", flush=True)
    zero_qkv_group = os.environ.get("UNIFOLM_WORLD_ZERO_QKV_GROUP", "32")
    if zero_qkv_group not in {"32", "1-large"}:
        raise ValueError("Invalid zero-low QKV group")
    if zero_qkv_group == "1-large":
        if os.environ.get("UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV", "0") != "all":
            raise ValueError("QKV traversal tuning requires all world zero-low QKV wrappers")
        grouped_qkv_count = sum(getattr(m, "_world_zero_weight_low_qkv_installed", False)
            for m in model.model.diffusion_model.modules())
        if grouped_qkv_count != 48:
            raise RuntimeError("Expected 48 world zero-low QKV wrappers")
        print("WORLD_ZERO_QKV_GROUP_MODEL mode=1-large installed=48 eligible_world_calls=35 "
              "retained_small_world_calls=3 group_large=1 group_small=32 "
              "scope=world_three_actual_large_shapes FP32_math_cache_and_stages=previous "
              "decision=previous", flush=True)
    geglu_c1280_subtile = os.environ.get("UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32", "0")
    if geglu_c1280_subtile not in {"0", "1"}:
        raise ValueError("Invalid C1280 GEGLU subtile mode")
    if geglu_c1280_subtile == "1":
        geglu_c1280_subtile_blocks = sum(bool(getattr(m, "_world_geglu_hidden_parts_c1280_zero_installed", False))
            for m in model.model.diffusion_model.modules())
        if geglu_c1280_subtile_blocks != 12:
            raise ValueError("C1280 subtile kernels require the twelve world zero-residual blocks")
        print("WORLD_GEGLU_C1280_SUBTILE32_MODEL mode=1 blocks=12 logical_BK=64 physical_BK=32 "
              "stages=2 BM=64 BN=32 group=32 scope=actual_M2560_640_world_only "
              "both_value_and_gate_short_sums=previous FP32_bias_GELU_and_cache=previous "
              "other_shapes_and_decision=previous", flush=True)
    geglu_c1280_tiles = os.environ.get("UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32_TILES", "0")
    if geglu_c1280_tiles not in {"0", "1"}:
        raise ValueError("Invalid C1280 GEGLU subtile tile mode")
    if geglu_c1280_tiles == "1":
        if os.environ.get("UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32", "0") != "1":
            raise ValueError("C1280 tile schedule requires the accepted subtile mode")
        if sum(bool(getattr(m, "_world_geglu_hidden_parts_c1280_zero_installed", False))
                for m in model.model.diffusion_model.modules()) != 12:
            raise ValueError("C1280 tile schedule requires the twelve world blocks")
        print("WORLD_GEGLU_C1280_SUBTILE32_TILES_MODEL mode=1 blocks=12 "
              "M2560_BM=64 M640_BM=32 BN=64 group=16 stages=2 "
              "logical_BK=64 physical_BK=32 inherited_subtile_marker=default_geometry "
              "other_shapes_and_decision=previous JIT_math_and_cache=previous", flush=True)
    decision_vendor_correction = os.environ.get("UNIFOLM_DECISION_VENDOR_CORRECTION", "0")
    if decision_vendor_correction not in {"0", "1"}:
        raise ValueError("Invalid decision vendor correction mode")
    if decision_vendor_correction == "1":
        if sum(bool(getattr(m, "_decision_chunked_hidden_parts_installed", False))
                for m in model.model.diffusion_model.modules()) != 10:
            raise ValueError("Vendor correction requires the ten decision chunk512 blocks")
        print("DECISION_VENDOR_CORRECTION_MODEL mode=1 blocks=10 M=10240 N=640 K=2560 chunk=512 "
              "first_two_products=original_cuBLAS third_product=device_predicated_Triton "
              "nonzero_path=complete_chunk_MMA weights=first_timed_warm_refresh_either_stage_same_count_storage "
              "cold_stale_predicate_and_other_shapes_backend=original_cuBLAS "
              "FP32_reduce_bias_residual=previous world=previous", flush=True)
    decision_paired_vendor = os.environ.get("UNIFOLM_DECISION_PAIRED_FUSED_VENDOR", "0")
    if decision_paired_vendor not in {"0", "1"}:
        raise ValueError("Invalid decision paired vendor mode")
    if decision_paired_vendor == "1":
        if os.environ.get("UNIFOLM_DECISION_VENDOR_CORRECTION", "0") != "1":
            raise ValueError("Paired vendor requires the accepted correction predicate")
        if sum(bool(getattr(m, "_decision_chunked_hidden_parts_installed", False))
                for m in model.model.diffusion_model.modules()) != 10:
            raise ValueError("Paired vendor requires the ten decision C640 blocks")
        print("DECISION_PAIRED_FUSED_VENDOR_MODEL mode=1 blocks=10 M=10240 N=640 K=2560 chunk=512 "
              "GEGLU_output=fresh_two_plane_allocation activation_copy=none "
              "first_two_products=single_row_paired_cuBLAS third_product_and_FP32_reduce_bias_residual=fused_device_predicate "
              "FP32_reduce_bias_residual=previous missing_layout_predicate_and_other_backend=previous "
              "weights=previous_first_timed_refresh world=previous", flush=True)
    packed_temporal_schedule = os.environ.get("UNIFOLM_WORLD_PACKED_TEMPORAL_SCHEDULE", "0")
    if packed_temporal_schedule not in {"0", "1"}:
        raise ValueError("Invalid packed temporal schedule")
    if packed_temporal_schedule == "1":
        count=sum(bool(getattr(m, "_world_zero_weight_low_temporal_installed", False))
            and getattr(m, "in_channels", None)==1280 for m in model.model.diffusion_model.modules())
        if count != 48:
            raise ValueError("Packed temporal schedule requires the 48 original C1280 wrappers")
        print("WORLD_PACKED_TEMPORAL_SCHEDULE_MODEL mode=1 blocks=48 C=1280 M2560_group=8 M640_group=32 "
              "stages=3 BM=64 BN=64 warps=4 logical_BK=64 "
              "JIT_math_device_predicate_packing_output_and_cache=previous narrow_and_decision=previous", flush=True)
    attention_projection_schedule = os.environ.get("UNIFOLM_WORLD_ATTENTION_PROJECTION_SCHEDULE", "0")
    if attention_projection_schedule not in {"0", "1"}:
        raise ValueError("Invalid attention projection schedule")
    if attention_projection_schedule == "1":
        count=sum(bool(getattr(m, "_world_attention_hidden_projection_installed", False))
            for m in model.model.diffusion_model.modules())
        if count != 10:
            raise ValueError("Attention projection schedule requires the ten original world wrappers")
        print("WORLD_ATTENTION_PROJECTION_SCHEDULE_MODEL mode=1 blocks=10 C320=5 C640=5 "
              "group=8 stages=3 BM=64 BN=64 BK=64 warps=4 "
              "JIT_math_cache_input_parts_and_FP32_output=previous decision_and_other_shapes=previous", flush=True)
    from .norm_presplit_temporal_runtime import requested as norm_presplit_requested
    norm_presplit_enabled=norm_presplit_requested()
    norm_presplit_count=0
    if norm_presplit_enabled:
        from .norm_presplit_temporal_runtime import FusedNormTemporalController
        controller=FusedNormTemporalController(model.model.diffusion_model,trace=False)
        norm_presplit_count=len(controller.selected)
        if norm_presplit_count!=40:raise ValueError('Unexpected norm/presplit temporal sequence count')
        controller.enabled=True
        model._norm_presplit_temporal_controller=controller
    print(f'NORM_PRESPLIT_TEMPORAL_MODEL mode={int(norm_presplit_enabled)} sequences={norm_presplit_count} '
          'scope=world_C320_C640 GN_stats=previous FP32_GN_SiLU_math=previous '
          'output=NCTHW_Half_high_low original_BK64_sums_and_predicates=true '
          'all_setup_inside_process_wall=true master_parameters=fp32 decision=previous',flush=True)
    from .norm2_qkv_pack import requested as norm2_qkv_requested
    norm2_qkv_enabled=norm2_qkv_requested()
    norm2_qkv_count=0
    if norm2_qkv_enabled:
        if (dataset!=STAGE_TF32_CASE or full_fp16_enabled or
                _stage_tf32_mode()!='decision-cudnn' or stage_tf32_ddim_prefix_steps()!=0):
            raise ValueError('Norm2 QKV packing requires accepted V2 world stage policy')
        from .norm2_qkv_pack import install as install_norm2_qkv_pack
        norm2_qkv_count=install_norm2_qkv_pack(model.model.diffusion_model)
    print(f'NORM2_QKV_PACK_MODEL mode={int(norm2_qkv_enabled)} blocks={norm2_qkv_count} '
          'scope=world_temporal_norm2 original_norm_GEMM_attention_JIT=true '
          'shared_original_QKV_cache=true dynamic_half_parts=each_call '
          'all_setup_inside_process_wall=true decision=previous',flush=True)
    condition_cache = os.environ.get("UNIFOLM_CONDITION_CACHE", "0")
    if condition_cache not in {"0", "1"}:
        raise ValueError("UNIFOLM_CONDITION_CACHE must be 0 or 1")
    fixed_cast_cache = os.environ.get("UNIFOLM_FIXED_CAST_CACHE", "0")
    if fixed_cast_cache not in {"0", "1"}:
        raise ValueError("UNIFOLM_FIXED_CAST_CACHE must be 0 or 1")
    if fixed_cast_cache == "1" and (condition_cache != "1" or not full_fp16_enabled):
        raise ValueError("Fixed cast cache requires FP16 denoiser and condition cache")
    model.model.diffusion_model._fixed_cast_cache_enabled = fixed_cast_cache == "1"
    print(f"FIXED_CAST_CACHE requested={fixed_cast_cache}", flush=True)
    if condition_cache == "1":
        diffusion = model.model.diffusion_model
        diffusion._condition_cache_enabled = True
        diffusion.action_unet._condition_cache_enabled = True
        diffusion.state_unet._condition_cache_enabled = True
        attention_count = 0
        for module in diffusion.modules():
            if type(module) is CrossAttention:
                module._condition_cache_enabled = True
                attention_count += 1
        print(f"CONDITION_CACHE enabled=true attention_modules={attention_count}",
              flush=True)
        def refresh_conditioning(args, kwargs):
            condition = args[4]
            cross_attention = condition["c_crossattn"]
            context = (cross_attention[0] if len(cross_attention) == 1 else
                       torch.cat(cross_attention, dim=1))
            action_condition = condition["c_crossattn_action"]
            if full_fp16_enabled:
                context, action_condition = diffusion._prepare_inference_condition_dtype(
                    context, action_condition)
            action_shape = args[1].shape[:2]
            diffusion.prepare_condition_cache(
                context, args[0].shape[2], action_condition, action_shape)
    else:
        refresh_conditioning = None
    graph_mode = os.environ.get("UNIFOLM_CUDA_GRAPH", "0")
    if graph_mode not in {"0", "1"}:
        raise ValueError("UNIFOLM_CUDA_GRAPH must be 0 or 1")
    if graph_mode == "1":
        from unifolm_wma.utils.cuda_graph import DenoiserGraph, prepare_graph_constants
        normalizers = prepare_graph_constants(model.model.diffusion_model)
        max_graphs = 4 if stage_tf32_ddim_prefix_steps() else 2
        skip_unchanged_inputs = os.environ.get(
            "UNIFOLM_GRAPH_SKIP_UNCHANGED_INPUTS", "0")
        if skip_unchanged_inputs not in {"0", "1"}:
            raise ValueError(
                "UNIFOLM_GRAPH_SKIP_UNCHANGED_INPUTS must be 0 or 1")
        model.apply_model = DenoiserGraph(
            model.apply_model, condition_refresher=refresh_conditioning,
            skip_unchanged_inputs=skip_unchanged_inputs == "1",
            max_graphs=max_graphs)
        print(f"CUDA_GRAPH enabled=true scope=complete_denoiser "
              f"cached_normalizers={normalizers} "
              f"skip_unchanged_inputs={skip_unchanged_inputs} "
              f"max_graphs={max_graphs}", flush=True)

    from unifolm_wma.utils.head_norm_mish_runtime import configure_head_fusion
    configure_head_fusion(model,mode=mode,dataset=dataset)

    from unifolm_wma.utils.serial_group_norm_runtime import configure_serial_norm
    configure_serial_norm(model,mode=mode,dataset=dataset)

    from unifolm_wma.utils.world_two_product_tiles_runtime import configure_two_product_tiles
    configure_two_product_tiles(model,mode=mode,dataset=dataset)

    from unifolm_wma.utils.norm_projection_pack_runtime import configure_norm_projection_pack
    configure_norm_projection_pack(model,mode=mode,dataset=dataset)

    from unifolm_wma.utils.out_projection_residual_runtime import configure_out_projection_residual
    configure_out_projection_residual(model,mode=mode,dataset=dataset)

    from unifolm_wma.utils.world_persistent_zero_low_runtime import configure_world_persistent_zero_low
    configure_world_persistent_zero_low(model,mode=mode,dataset=dataset)
