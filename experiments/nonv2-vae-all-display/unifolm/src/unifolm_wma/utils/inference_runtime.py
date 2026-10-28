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
    from .uninitialized_memory import configure_empty_fill
    configure_empty_fill()
    full_fp16_mode = os.environ.get("UNIFOLM_FULL_DENOISER_FP16", "0")
    full_fp16_enabled = full_denoiser_fp16_enabled(full_fp16_mode, dataset)
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
            protected_feedback_ids = convert_half_with_fp32_feedback(diffusion)
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
    if full_fp16_enabled and tf32x3_linears != "0":
        raise ValueError("TF32x3 linears require FP32 denoiser parameters")
    if enabled or tf32x3_linears != "0":
        if torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("TF32x3 requires FP32 outside the selected operators")
        if tf32x3_linears == "triton":
            if enabled:
                raise ValueError("Triton compensated linears require native convolutions")
            from unifolm_wma.utils.triton_compensated_linear import install_compensated_linears
            counts = install_compensated_linears(model.model.diffusion_model)
        else:
            from unifolm_wma.utils.tf32x3 import install_tf32x3
            counts = install_tf32x3(model.model.diffusion_model,
                                    convolutions=enabled)
    else:
        counts = {}
    print(f"DENOISER_MODE requested={mode} case={dataset} wrapped={counts}", flush=True)
    print(f"TF32X3_LINEARS requested={tf32x3_linears} "
          f"convolutions={str(enabled).lower()} wrapped={counts}", flush=True)
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
                    temporal_conv2d_count += 1
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
    fused_temporal_attention = os.environ.get("UNIFOLM_FUSED_TEMPORAL_ATTENTION", "0")
    if fused_temporal_attention not in {"0", "1"}:
        raise ValueError("UNIFOLM_FUSED_TEMPORAL_ATTENTION must be 0 or 1")
    for module in model.model.diffusion_model.modules():
        if isinstance(module, CrossAttention):
            module._inference_fused_temporal_attention = fused_temporal_attention == "1"
    print(f"FUSED_TEMPORAL_ATTENTION requested={fused_temporal_attention}",flush=True)
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
