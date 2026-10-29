import argparse, os
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

# Set before importing the heavy model stack so the optional profile includes
# import time as part of the cold-start total.
PROCESS_START_TIME = time.perf_counter()

# cuBLAS reads this before CUDA is initialized.  The launcher sets the same
# value; keeping this fallback makes direct ``python`` invocation deterministic
# for PyTorch operations as well.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import torch
from guard_crossstep211 import announce_worker as _announce_crossstep_worker
_crossstep_guard_hold = _announce_crossstep_worker()
import torchvision
import numpy as np
import logging

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
from torch import nn
from eval_utils import populate_queues, log_to_tensorboard
from collections import deque
from torch import Tensor

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.utils.utils import instantiate_from_config
from unifolm_wma.utils.inference_runtime import (
    TF32_CANDIDATE_CASES,
    full_denoiser_fp16_enabled,
    generation_math,
    skip_parameter_initialization,
)


class TimingCollector:
    """Collect wall-clock and optional CUDA-event timings for inference phases."""

    def __init__(self, process_start_time: float):
        self.process_start_time = process_start_time
        self.stages = defaultdict(lambda: {
            "wall_s": 0.0,
            "cuda_s": 0.0,
            "count": 0,
        })

    @contextmanager
    def stage(self, name: str, cuda: bool = False):
        use_cuda_events = cuda and torch.cuda.is_available()
        if use_cuda_events:
            # Do not charge work queued by an earlier phase to this phase.
            torch.cuda.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()

        wall_start = time.perf_counter()
        try:
            yield
        finally:
            cuda_s = 0.0
            if use_cuda_events:
                cuda_end.record()
                torch.cuda.synchronize()
                cuda_s = cuda_start.elapsed_time(cuda_end) / 1000.0

            record = self.stages[name]
            record["wall_s"] += time.perf_counter() - wall_start
            record["cuda_s"] += cuda_s
            record["count"] += 1

    def write(self, path: str | None) -> None:
        if path is None:
            return
        profile_dir = os.path.dirname(path)
        if profile_dir:
            os.makedirs(profile_dir, exist_ok=True)

        stages = {
            name: {
                "wall_s": round(values["wall_s"], 6),
                "cuda_s": round(values["cuda_s"], 6),
                "count": values["count"],
            }
            for name, values in sorted(self.stages.items())
        }
        profiled_wall_s = sum(values["wall_s"] for values in self.stages.values())
        elapsed_s = time.perf_counter() - self.process_start_time
        payload = {
            "schema_version": 1,
            "total_process_wall_s": round(elapsed_s, 6),
            "profiled_wall_s": round(profiled_wall_s, 6),
            "unprofiled_wall_s": round(max(0.0, elapsed_s - profiled_wall_s),
                                        6),
            "stages": stages,
        }
        with open(path, "w", encoding="utf-8") as profile_file:
            json.dump(payload, profile_file, indent=2, sort_keys=True)
            profile_file.write("\n")

    def print_compute_summary(self) -> None:
        """Print the complete model-compute interval, excluding all I/O.

        The two generation stages include the conditioning encoders, DDIM
        denoising and VAE decoding. They intentionally exclude checkpoint and
        input loading, queue/input preparation, TensorBoard, and MP4 writes.
        CUDA events are authoritative on GPU; wall time is a CPU fallback.
        """
        decision = self.stages.get("decision_generation", {})
        simulation = self.stages.get("simulation_generation", {})
        decision_cuda = decision.get("cuda_s", 0.0)
        simulation_cuda = simulation.get("cuda_s", 0.0)
        decision_wall = decision.get("wall_s", 0.0)
        simulation_wall = simulation.get("wall_s", 0.0)
        total_cuda = decision_cuda + simulation_cuda
        total_wall = decision_wall + simulation_wall
        calls = decision.get("count", 0) + simulation.get("count", 0)

        if torch.cuda.is_available():
            print(
                ">>> Pure model compute time (CUDA; excludes load/save): "
                f"{total_cuda:.3f}s "
                f"[decision {decision_cuda:.3f}s + world {simulation_cuda:.3f}s; "
                f"{calls} generation calls]"
            )
        else:
            print(
                ">>> Pure model compute time (wall; excludes load/save): "
                f"{total_wall:.3f}s "
                f"[decision {decision_wall:.3f}s + world {simulation_wall:.3f}s; "
                f"{calls} generation calls]"
            )


@contextmanager
def operator_profile(stage: str, iteration: int):
    """Profile one steady-state generation call when explicitly requested."""
    enabled = os.environ.get("UNIFOLM_OPERATOR_PROFILE", "0")
    if enabled not in {"0", "1"}:
        raise ValueError("UNIFOLM_OPERATOR_PROFILE must be 0 or 1")
    target_iteration = int(os.environ.get(
        "UNIFOLM_OPERATOR_PROFILE_ITER", "1"))
    if enabled != "1" or iteration != target_iteration:
        yield
        return

    profile_dir = os.environ.get("UNIFOLM_OPERATOR_PROFILE_DIR", "profiles")
    os.makedirs(profile_dir, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False) as profiler:
        yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stem = os.path.join(profile_dir, f"{stage}-iteration-{iteration}")
    profiler.export_chrome_trace(f"{stem}.json")
    with open(f"{stem}.txt", "w", encoding="utf-8") as handle:
        handle.write(profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=100))
        handle.write("\n")
    print(f"OPERATOR_PROFILE stage={stage} iteration={iteration} path={stem}",
          flush=True)


def configure_reproducibility(seed: int) -> None:
    """Configure deterministic single-process inference before model loading.

    The world-model sampler, VAE encoder, and DDIM scheduler all draw random
    tensors.  Setting every RNG here, together with the launcher's cuBLAS
    environment, makes a fixed non-negative ``--seed`` reproduce the same
    inference trajectory on the same software stack and GPU architecture.
    """
    if seed < 0:
        raise ValueError(
            "Deterministic inference requires --seed to be a non-negative integer."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Disable autotuning and TF32 because both can choose numerically different
    # CUDA kernels between launches.  Raise instead of silently accepting a
    # known non-deterministic PyTorch operation.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    # Preserve Lightning's integration and seed any worker created by a future
    # data-pipeline change.  The explicit seeds above keep this script
    # independent of Lightning's implementation details.
    seed_everything(seed, workers=True)


INPUT_CACHE_SCHEMA_VERSION = 1


def sha256_file(path: str) -> str:
    """Return a content digest used to reject caches for another config."""
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_cache_path(cache_dir: str, dataset: str, frame_stride: int) -> str:
    """Return the stable input-cache name for one benchmark case/stride."""
    return os.path.join(cache_dir, f"{dataset}-fs{frame_stride}.pt")


def capture_random_state() -> dict:
    """Capture every RNG that can affect a future sampling trajectory.

    This deliberately differs from the TensorRT helper: ``default_rng()`` is
    not included because creating a new NumPy Generator does not capture an
    application's generator.  CUDA keeps a generator per visible device, so
    all of them are saved rather than only the current device.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    else:
        cuda_states = []
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda_all": cuda_states,
        "cuda_device_count": len(cuda_states),
    }


def restore_random_state(state: dict) -> None:
    """Restore an input snapshot's sampling point exactly.

    A cache is intentionally tied to the visible CUDA-device count.  Restoring
    one generator onto a different device topology makes a benchmark appear
    reproducible while silently changing the random stream, so fail loudly.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state["torch_cuda_all"]
    if cuda_states:
        available = torch.cuda.device_count()
        if available != state["cuda_device_count"]:
            raise RuntimeError(
                "Input cache CUDA RNG state was captured with "
                f"{state['cuda_device_count']} visible device(s), but this run "
                f"has {available}. Set CUDA_VISIBLE_DEVICES consistently."
            )
        torch.cuda.set_rng_state_all([saved.cpu() for saved in cuda_states])


def _cpu_tensor(value: Tensor) -> Tensor:
    """Detach a cache payload from autograd and any accelerator allocation."""
    return value.detach().cpu().contiguous().clone()


def validate_input_cache(cache: dict,
                         args: argparse.Namespace,
                         frame_stride: int,
                         model: nn.Module) -> None:
    """Fail rather than benchmarking stale preprocessing or sampler inputs."""
    expected = {
        "schema_version": INPUT_CACHE_SCHEMA_VERSION,
        "dataset": args.dataset,
        "frame_stride": frame_stride,
        "config_sha256": sha256_file(args.config),
        "video_length": args.video_length,
        "height": args.height,
        "width": args.width,
        "n_obs_steps_imagen": model.n_obs_steps_imagen,
    }
    for key, value in expected.items():
        if cache.get(key) != value:
            raise RuntimeError(
                f"Input cache {key!r} is {cache.get(key)!r}, expected {value!r}. "
                "Rebuild it with tools/cache_builder/prepare_compute_benchmark_inputs.sh."
            )

    required = {"sample", "batch", "ori_state_dim", "ori_action_dim",
                "model_input_fs", "rng_state"}
    missing = required.difference(cache)
    if missing:
        raise RuntimeError(f"Input cache is missing fields: {sorted(missing)}")


def load_input_cache(cache_path: str,
                     args: argparse.Namespace,
                     frame_stride: int,
                     model: nn.Module) -> dict:
    """Load one trusted local benchmark cache and validate its provenance."""
    # This cache is generated locally by this script and includes Python and
    # NumPy RNG objects, so it cannot use weights_only=True.  Do not load an
    # untrusted .pt file.
    cache = torch.load(cache_path, map_location="cpu")
    validate_input_cache(cache, args, frame_stride, model)
    return cache


def write_input_cache(cache_path: str,
                      args: argparse.Namespace,
                      frame_stride: int,
                      model: nn.Module,
                      sample: dict,
                      batch: dict[str, Tensor],
                      ori_state_dim: int,
                      ori_action_dim: int,
                      model_input_fs: float) -> None:
    """Persist initial preprocessed inputs and the pre-sampler RNG checkpoint."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = {
        "schema_version": INPUT_CACHE_SCHEMA_VERSION,
        "dataset": args.dataset,
        "frame_stride": frame_stride,
        "config_sha256": sha256_file(args.config),
        "video_length": args.video_length,
        "height": args.height,
        "width": args.width,
        "n_obs_steps_imagen": model.n_obs_steps_imagen,
        "sample": {
            "videoid": int(sample["videoid"]),
            "fps": float(sample["fps"]),
            "instruction": str(sample["instruction"]),
        },
        "batch": {
            "observation.image": _cpu_tensor(batch["observation.image"]),
            "observation.state": _cpu_tensor(batch["observation.state"]),
            "action": _cpu_tensor(batch["action"]),
        },
        "ori_state_dim": int(ori_state_dim),
        "ori_action_dim": int(ori_action_dim),
        "model_input_fs": model_input_fs,
        # Captured after all initialization/preprocessing and immediately
        # before the first decision sampler call.
        "rng_state": capture_random_state(),
    }
    temporary_path = f"{cache_path}.tmp-{os.getpid()}"
    torch.save(cache, temporary_path)
    os.replace(temporary_path, cache_path)


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Args:
        module (nn.Module): The model whose device is to be inferred.

    Returns:
        torch.device: The device of the model's parameters.
    """
    return next(iter(module.parameters())).device


@contextmanager
def preserve_module_registration_on_assign(enabled: bool):
    """Keep parameter/buffer registration while assigning checkpoint storage."""
    if not enabled:
        yield
        return
    previous = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(True)
    try:
        yield
    finally:
        torch.__future__.set_swap_module_params_on_conversion(previous)


def load_model_checkpoint(model: nn.Module,
                          ckpt: str,
                          timing: TimingCollector | None = None) -> nn.Module:
    """Load model weights from checkpoint file.

    Args:
        model (nn.Module): Model instance.
        ckpt (str): Path to the checkpoint file.

    Returns:
        nn.Module: Model with loaded weights.
    """
    mmap_mode = os.environ.get("UNIFOLM_CHECKPOINT_MMAP", "0")
    assign_mode = os.environ.get("UNIFOLM_CHECKPOINT_ASSIGN", "0")
    if mmap_mode not in {"0", "1"}:
        raise ValueError("UNIFOLM_CHECKPOINT_MMAP must be 0 or 1")
    if assign_mode not in {"0", "1"}:
        raise ValueError("UNIFOLM_CHECKPOINT_ASSIGN must be 0 or 1")
    mmap_enabled = mmap_mode == "1"
    assign_enabled = assign_mode == "1"
    print(f"CHECKPOINT_LOAD mmap={str(mmap_enabled).lower()} "
          f"assign={str(assign_enabled).lower()}", flush=True)
    with (timing.stage("checkpoint_deserialize") if timing else nullcontext()):
        state_dict = torch.load(
            ckpt, map_location="cpu", mmap=mmap_enabled)
    with (timing.stage("checkpoint_load_state_dict") if timing else
          nullcontext()), preserve_module_registration_on_assign(assign_enabled):
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
            try:
                model.load_state_dict(
                    state_dict, strict=True, assign=assign_enabled)
            except:
                new_pl_sd = OrderedDict()
                for k, v in state_dict.items():
                    new_pl_sd[k] = v

                for k in list(new_pl_sd.keys()):
                    if "framestride_embed" in k:
                        new_key = k.replace("framestride_embed",
                                            "fps_embedding")
                        new_pl_sd[new_key] = new_pl_sd[k]
                        del new_pl_sd[k]
                model.load_state_dict(
                    new_pl_sd, strict=True, assign=assign_enabled)
        else:
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]] = state_dict['module'][key]
            model.load_state_dict(new_pl_sd, assign=assign_enabled)
    print('>>> model checkpoint loaded.')
    return model


def video_to_frames(video: Tensor) -> Tensor:
    """Convert a (B, C, T, H, W) video to uint8 (T, H, W, C) frames."""
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    if n == 1:
        # make_grid(single_image, padding=0) returns that image unchanged.
        # Avoid the per-frame Python loop for the fixed benchmark batch size.
        grid = video[0].permute(1, 0, 2, 3)
    else:
        video = video.permute(2, 0, 1, 3, 4)
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0)
            for framesheet in video
        ]
        grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    return (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)


def write_video_frames(frames: Tensor, filename: str, fps: int) -> None:
    """Encode uint8 (T, H, W, C) frames using the benchmark's settings."""
    torchvision.io.write_video(filename,
                               frames,
                               fps=fps,
                               video_codec='h264',
                               options={'crf': '10'})


def save_results(video: Tensor, filename: str, fps: int = 8) -> None:
    """Save video tensor to file using torchvision."""
    write_video_frames(video_to_frames(video), filename, fps)


class VideoWriter:
    """Serialize MP4 writes, optionally overlapping them with GPU inference."""

    def __init__(self, asynchronous: bool, stream_full_video: bool = False):
        if stream_full_video and not asynchronous:
            raise ValueError(
                "Streaming full video requires asynchronous video writing")
        self.executor = (ThreadPoolExecutor(max_workers=1)
                         if asynchronous else None)
        self.stream_full_video = stream_full_video
        self.futures = []
        self.full_container = None
        self.full_stream = None
        self.full_filename = None

    def write(self, video: Tensor, filename: str, fps: int) -> Tensor:
        cpu_video = video.detach().cpu()
        if self.executor is None:
            save_results(cpu_video, filename, fps=fps)
        else:
            self.futures.append(
                self.executor.submit(save_results, cpu_video, filename, fps))
        return cpu_video

    def write_world(self, video: Tensor, filename: str, full_filename: str,
                    full_frame_count: int, fps: int) -> Tensor:
        """Write one WM segment and feed the same raw frames to the full MP4."""
        if not self.stream_full_video:
            return self.write(video, filename, fps)
        cpu_video = video.detach().cpu()
        self.futures.append(
            self.executor.submit(self._write_world_and_append, cpu_video,
                                 filename, full_filename, full_frame_count,
                                 fps))
        return cpu_video

    def _write_world_and_append(self, video: Tensor, filename: str,
                                full_filename: str, full_frame_count: int,
                                fps: int) -> None:
        frames = video_to_frames(video)
        write_video_frames(frames, filename, fps)
        self._append_full_frames(frames[:full_frame_count], full_filename, fps)

    def _append_full_frames(self, frames: Tensor, filename: str,
                            fps: int) -> None:
        if self.full_filename != filename:
            self._close_full_stream()
            import av
            self.full_container = av.open(filename, mode="w")
            self.full_stream = self.full_container.add_stream("h264", rate=fps)
            self.full_stream.width = frames.shape[2]
            self.full_stream.height = frames.shape[1]
            self.full_stream.pix_fmt = "yuv420p"
            self.full_stream.options = {'crf': '10'}
            self.full_filename = filename
        import av
        for image in frames.numpy():
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pict_type = "NONE"
            for packet in self.full_stream.encode(frame):
                self.full_container.mux(packet)

    def _close_full_stream(self) -> None:
        if self.full_container is None:
            return
        for packet in self.full_stream.encode():
            self.full_container.mux(packet)
        self.full_container.close()
        self.full_container = None
        self.full_stream = None
        self.full_filename = None

    def close(self) -> None:
        if self.executor is None:
            return
        if self.stream_full_video:
            self.futures.append(self.executor.submit(self._close_full_stream))
        self.executor.shutdown(wait=True)
        for future in self.futures:
            future.result()


def get_init_frame_path(data_dir: str, sample: dict) -> str:
    """Construct the init_frame path from directory and sample metadata.

    Args:
        data_dir (str): Base directory containing videos.
        sample (dict): Dictionary containing 'data_dir' and 'videoid'.

    Returns:
        str: Full path to the video file.
    """
    rel_video_fp = os.path.join(sample['data_dir'],
                                str(sample['videoid']) + '.png')
    full_image_fp = os.path.join(data_dir, 'images', rel_video_fp)
    return full_image_fp


def get_transition_path(data_dir: str, sample: dict) -> str:
    """Construct the full transition file path from directory and sample metadata.

    Args:
        data_dir (str): Base directory containing transition files.
        sample (dict): Dictionary containing 'data_dir' and 'videoid'.

    Returns:
        str: Full path to the HDF5 transition file.
    """
    rel_transition_fp = os.path.join(sample['data_dir'],
                                     str(sample['videoid']) + '.h5')
    full_transition_fp = os.path.join(data_dir, 'transitions',
                                      rel_transition_fp)
    return full_transition_fp


def prepare_init_input(start_idx: int,
                       init_frame_path: str,
                       transition_dict: dict[str, torch.Tensor],
                       frame_stride: int,
                       wma_data,
                       video_length: int = 16,
                       n_obs_steps: int = 2) -> dict[str, Tensor]:
    """
    Extracts a structured sample from a video sequence including frames, states, and actions,
    along with properly padded observations and pre-processed tensors for model input.

    Args:
        start_idx (int): Starting frame index for the current clip.
        video: decord video instance.
        transition_dict (Dict[str, Tensor]): Dictionary containing tensors for 'action', 
                                             'observation.state', 'action_type', 'state_type'.
        frame_stride (int): Temporal stride between sampled frames.
        wma_data: Object that holds configuration and utility functions like normalization, 
                transformation, and resolution info.
        video_length (int, optional): Number of frames to sample from the video. Default is 16.
        n_obs_steps (int, optional): Number of historical steps for observations. Default is 2.
    """

    # Raw-image decoding is needed only by the organizer's cache builder.
    # The competition path loads the already transformed tensor from a cache.
    from PIL import Image

    indices = [start_idx + frame_stride * i for i in range(video_length)]
    init_frame = Image.open(init_frame_path).convert('RGB')
    init_frame = torch.tensor(np.array(init_frame)).unsqueeze(0).permute(
        3, 0, 1, 2).float()

    if start_idx < n_obs_steps - 1:
        state_indices = list(range(0, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]
        num_padding = n_obs_steps - 1 - start_idx
        first_slice = states[0:1, :]  # (t, d)
        padding = first_slice.repeat(num_padding, 1)
        states = torch.cat((padding, states), dim=0)
    else:
        state_indices = list(range(start_idx - n_obs_steps + 1, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]

    actions = transition_dict['action'][indices, :]

    ori_state_dim = states.shape[-1]
    ori_action_dim = actions.shape[-1]

    frames_action_state_dict = {
        'action': actions,
        'observation.state': states,
    }
    frames_action_state_dict = wma_data.normalizer(frames_action_state_dict)
    frames_action_state_dict = wma_data.get_uni_vec(
        frames_action_state_dict,
        transition_dict['action_type'],
        transition_dict['state_type'],
    )

    if wma_data.spatial_transform is not None:
        init_frame = wma_data.spatial_transform(init_frame)
    init_frame = (init_frame / 255 - 0.5) * 2

    data = {
        'observation.image': init_frame,
    }
    data.update(frames_action_state_dict)
    return data, ori_state_dim, ori_action_dim


def get_latent_z(model, videos: Tensor) -> Tensor:
    """
    Extracts latent features from a video batch using the model's first-stage encoder.

    Args:
        model: the world model.
        videos (Tensor): Input videos of shape [B, C, T, H, W].

    Returns:
        Tensor: Latent video tensor of shape [B, C, T, H, W].
    """
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


@torch.no_grad()
def image_guided_synthesis_sim_mode(
        model: torch.nn.Module,
        prompts: list[str],
        observation: dict,
        noise_shape: tuple[int, int, int, int, int],
        action_cond_step: int = 16,
        n_samples: int = 1,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 1.0,
        fs: int | None = None,
        text_input: bool = True,
        timestep_spacing: str = 'uniform',
        guidance_rescale: float = 0.0,
        sim_mode: bool = True,
        generation_stage: str | None = None,
        dataset: str | None = None,
        **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs image-guided video generation in a simulation-style mode with optional multimodal guidance (image, state, action, text).

    Args:
        model (torch.nn.Module): The diffusion-based generative model with multimodal conditioning.
        prompts (list[str]): A list of textual prompts to guide the synthesis process.
        observation (dict): A dictionary containing observed inputs including:
            - 'observation.images.top': Tensor of shape [B, O, C, H, W] (top-down images)
            - 'observation.state': Tensor of shape [B, O, D] (state vector)
            - 'action': Tensor of shape [B, T, D] (action sequence)
        noise_shape (tuple[int, int, int, int, int]): Shape of the latent variable to generate, 
            typically (B, C, T, H, W).
        action_cond_step (int): Number of time steps where action conditioning is applied. Default is 16.
        n_samples (int): Number of samples to generate (unused here, always generates 1). Default is 1.
        ddim_steps (int): Number of DDIM sampling steps. Default is 50.
        ddim_eta (float): DDIM eta parameter controlling the stochasticity. Default is 1.0.
        unconditional_guidance_scale (float): Scale for classifier-free guidance. If 1.0, guidance is off.
        fs (int | None): Frame index to condition on, broadcasted across the batch if specified. Default is None.
        text_input (bool): Whether to use text prompt as conditioning. If False, uses empty strings. Default is True.
        timestep_spacing (str): Timestep sampling method in DDIM sampler. Typically "uniform" or "linspace".
        guidance_rescale (float): Guidance rescaling factor to mitigate overexposure from classifier-free guidance.
        sim_mode (bool): Whether to perform world-model interaction or decision-making using the world-model.
        generation_stage (str | None): Optional decision/simulation label for a
            stage-specific denoiser math policy.
        dataset (str | None): Dataset name used to scope that math policy.
        **kwargs: Additional arguments passed to the DDIM sampler.

    Returns:
        batch_variants (torch.Tensor): Predicted pixel-space video frames [B, C, T, H, W].
        actions (torch.Tensor): Predicted action sequences [B, T, D] from diffusion decoding.
        states (torch.Tensor): Predicted state sequences [B, T, D] from diffusion decoding.
    """
    b, _, t, _, _ = noise_shape
    cache_ddim_schedule = os.environ.get("UNIFOLM_DDIM_SCHEDULE_CACHE",
                                         "0") == "1"
    ddim_sampler = (getattr(model, "_inference_ddim_sampler", None)
                    if cache_ddim_schedule else None)
    if ddim_sampler is None:
        ddim_sampler = DDIMSampler(model)
        if cache_ddim_schedule:
            model._inference_ddim_sampler = ddim_sampler
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    img = observation['observation.images.top'].permute(0, 2, 1, 3, 4)
    cond_img = rearrange(img, 'b o c h w -> (b o) c h w')[-1:]
    cond_img_emb = model.embedder(cond_img)
    cond_img_emb = model.image_proj_model(cond_img_emb)

    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, img.permute(0, 2, 1, 3, 4))
        img_cat_cond = z[:, :, -1:, :, :]
        img_cat_cond = repeat(img_cat_cond,
                              'b c t h w -> b c (repeat t) h w',
                              repeat=noise_shape[2])
        cond = {"c_concat": [img_cat_cond]}

    if not text_input:
        prompts = [""] * batch_size
    cond_ins_emb = model.get_learned_conditioning(prompts)

    cond_state_emb = model.state_projector(observation['observation.state'])
    cond_state_emb = cond_state_emb + model.agent_state_pos_emb

    skip_unused_action = (
        os.environ.get("UNIFOLM_SKIP_UNUSED_ACTION_PROJECTION", "0") == "1")
    if not sim_mode and skip_unused_action:
        action_steps = observation['action'].shape[1]
        cond_action_emb = torch.zeros_like(
            model.agent_action_pos_emb[:, :action_steps].expand(
                batch_size, -1, -1))
    else:
        cond_action_emb = model.action_projector(observation['action'])
        cond_action_emb = cond_action_emb + model.agent_action_pos_emb
        if not sim_mode:
            cond_action_emb = torch.zeros_like(cond_action_emb)

    cond["c_crossattn"] = [
        torch.cat(
            [cond_state_emb, cond_action_emb, cond_ins_emb, cond_img_emb],
            dim=1)
    ]
    cond["c_crossattn_action"] = [
        observation['observation.images.top'][:, :,
                                              -model.n_obs_steps_acting:],
        observation['observation.state'][:, -model.n_obs_steps_acting:],
        sim_mode,
        False,
    ]

    uc = None
    kwargs.update({"unconditional_conditioning_img_nonetext": None})
    cond_mask = None
    cond_z0 = None
    if ddim_sampler is not None:
        full_fp16_mode = os.environ.get("UNIFOLM_FULL_DENOISER_FP16", "0")
        full_fp16_enabled = full_denoiser_fp16_enabled(
            full_fp16_mode, dataset)
        math_context = (generation_math(generation_stage, dataset)
                        if generation_stage is not None and dataset is not None
                        else nullcontext())
        with math_context:
            samples, actions, states, intermedia = ddim_sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=batch_size,
                shape=noise_shape[1:],
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                cfg_img=None,
                mask=cond_mask,
                x0=cond_z0,
                fs=fs,
                timestep_spacing=timestep_spacing,
                guidance_rescale=guidance_rescale,
                precision=16 if full_fp16_enabled else None,
                inference_stage=generation_stage,
                inference_dataset=dataset,
                **kwargs)

        if full_fp16_enabled:
            samples = samples.float()
            actions = actions.float()
            states = states.float()

        # Reconstruct from latent to pixel space
        batch_decode_mode = os.environ.get(
            "UNIFOLM_BATCH_DECISION_DECODE", "0")
        batch_decision_decode = (
            generation_stage == "decision" and
            (batch_decode_mode == "1" or
             (batch_decode_mode == "selective" and dataset in
              TF32_CANDIDATE_CASES)))
        original_perframe_ae = model.perframe_ae
        if batch_decision_decode:
            model.perframe_ae = False
        try:
            if os.environ.get("UNIFOLM_WORLD_DECODE_PROTECTED_TF32_CHUNK", "0") != "0":
                if os.environ.get("UNIFOLM_WORLD_DECODE_NONFEEDBACK_CACHE", "0") != "0":
                    if os.environ.get("UNIFOLM_WORLD_DECODE_NONFEEDBACK_OVERLAP","0")!="0":
                        from unifolm_wma.utils.vae_nonfeedback_overlap_runtime import decode_for_stage
                    else:
                        from unifolm_wma.utils.vae_nonfeedback_cache_runtime import decode_for_stage
                else:
                    from unifolm_wma.utils.vae_protected_tf32_runtime import decode_for_stage
                batch_images = decode_for_stage(
                    model, samples, dataset=dataset, stage=generation_stage,
                    execution_steps=action_cond_step)
            else:
                batch_images = model.decode_first_stage(samples)
        finally:
            model.perframe_ae = original_perframe_ae
        batch_variants = batch_images

    return batch_variants, actions, states


def run_inference(args: argparse.Namespace,
                  timing: TimingCollector | None = None) -> None:
    """
    Run inference pipeline on prompts and image inputs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        None
    """
    tensorboard_video = os.environ.get("UNIFOLM_TENSORBOARD_VIDEO", "1")
    if tensorboard_video not in {"0", "1"}:
        raise ValueError("UNIFOLM_TENSORBOARD_VIDEO must be 0 or 1")
    tensorboard_video = tensorboard_video == "1"
    print(f"TENSORBOARD_VIDEO enabled={str(tensorboard_video).lower()}",
          flush=True)
    async_video = os.environ.get("UNIFOLM_ASYNC_VIDEO", "0")
    if async_video not in {"0", "1"}:
        raise ValueError("UNIFOLM_ASYNC_VIDEO must be 0 or 1")
    async_video = async_video == "1"
    print(f"ASYNC_VIDEO enabled={str(async_video).lower()}", flush=True)
    stream_full_video = os.environ.get("UNIFOLM_STREAM_FULL_VIDEO", "0")
    if stream_full_video not in {"0", "1"}:
        raise ValueError("UNIFOLM_STREAM_FULL_VIDEO must be 0 or 1")
    stream_full_video = stream_full_video == "1"
    print(f"STREAM_FULL_VIDEO enabled={str(stream_full_video).lower()}",
          flush=True)
    video_writer = VideoWriter(async_video, stream_full_video)
    ddim_schedule_cache = os.environ.get("UNIFOLM_DDIM_SCHEDULE_CACHE", "0")
    if ddim_schedule_cache not in {"0", "1"}:
        raise ValueError("UNIFOLM_DDIM_SCHEDULE_CACHE must be 0 or 1")
    print(f"DDIM_SCHEDULE_CACHE enabled={str(ddim_schedule_cache == '1').lower()}",
          flush=True)
    skip_unused_action = os.environ.get(
        "UNIFOLM_SKIP_UNUSED_ACTION_PROJECTION", "0")
    if skip_unused_action not in {"0", "1"}:
        raise ValueError(
            "UNIFOLM_SKIP_UNUSED_ACTION_PROJECTION must be 0 or 1")
    print(
        "SKIP_UNUSED_ACTION_PROJECTION enabled="
        f"{str(skip_unused_action == '1').lower()}",
        flush=True)
    batch_decision_decode = os.environ.get(
        "UNIFOLM_BATCH_DECISION_DECODE", "0")
    if batch_decision_decode not in {"0", "1", "selective"}:
        raise ValueError(
            "UNIFOLM_BATCH_DECISION_DECODE must be 0, 1, or selective")
    effective_batch_decision_decode = (
        batch_decision_decode == "1" or
        (batch_decision_decode == "selective" and args.dataset in
         TF32_CANDIDATE_CASES))
    print(
        f"BATCH_DECISION_DECODE requested={batch_decision_decode} "
        f"enabled={str(effective_batch_decision_decode).lower()}",
        flush=True)

    # The benchmark validates the standalone MP4 files below. TensorBoard
    # embeds a second encoded copy for visualization and is optional.
    with (timing.stage("output_setup") if timing else nullcontext()):
        os.makedirs(args.savedir + '/inference', exist_ok=True)
        if tensorboard_video:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = args.savedir + f"/tensorboard"
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

    # Discover caches before loading the source prompt.  A complete cache hit
    # avoids CSV, H5, PNG, normalization-stat and WMAData startup entirely.
    cache_paths = {}
    if args.input_cache_dir is not None:
        for fs in args.frame_stride:
            cache_path = input_cache_path(args.input_cache_dir, args.dataset,
                                          fs)
            # Explicit materialization always follows the source path so a
            # changed preprocessing implementation can intentionally replace
            # an existing snapshot.
            if os.path.exists(cache_path) and not args.write_input_cache:
                cache_paths[fs] = cache_path
            elif args.require_input_cache:
                raise FileNotFoundError(
                    f"Required input cache does not exist: {cache_path}. "
                    "Run tools/cache_builder/prepare_compute_benchmark_inputs.sh first.")

    all_input_caches_available = len(cache_paths) == len(args.frame_stride)
    df = None
    if not all_input_caches_available:
        with (timing.stage("prompt_csv_read") if timing else nullcontext()):
            import pandas as pd
            csv_path = os.path.join(args.prompt_dir, f"{args.dataset}.csv")
            df = pd.read_csv(csv_path)

    # Load config
    with (timing.stage("config_and_model_construct") if timing else
          nullcontext()):
        config = OmegaConf.load(args.config)
        if args.data_dir is not None:
            config['data']['params']['test']['params'][
                'data_dir'] = args.data_dir
        config['model']['params']['wma_config']['params'][
            'use_checkpoint'] = False
        skip_init = os.environ.get("UNIFOLM_SKIP_INIT", "0")
        if skip_init not in {"0", "1"}:
            raise ValueError("UNIFOLM_SKIP_INIT must be 0 or 1")
        skip_init = skip_init == "1"
        meta_init = os.environ.get("UNIFOLM_META_INIT", "0")
        if meta_init not in {"0", "1"}:
            raise ValueError("UNIFOLM_META_INIT must be 0 or 1")
        meta_init = meta_init == "1"
        if (meta_init and
                os.environ.get("UNIFOLM_CHECKPOINT_ASSIGN", "0") != "1"):
            raise ValueError(
                "UNIFOLM_META_INIT requires UNIFOLM_CHECKPOINT_ASSIGN=1")
        construction_device = (torch.device("meta")
                               if meta_init else nullcontext())
        with skip_parameter_initialization(skip_init), construction_device:
            model = instantiate_from_config(config.model)
        print(f"PARAMETER_INITIALIZATION skipped={str(skip_init).lower()}",
              flush=True)
        print(f"META_INITIALIZATION enabled={str(meta_init).lower()}",
              flush=True)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path, timing)
    if meta_init:
        remaining_meta = [
            name for name, value in list(model.named_parameters()) +
            list(model.named_buffers()) if value.is_meta
        ]
        if remaining_meta:
            raise RuntimeError(
                f"Checkpoint left meta tensors unresolved: {remaining_meta}")
    model.eval()
    print(f'>>> Load pre-trained model ...')

    # The cache already contains the output of WMAData's normalizer and
    # spatial transform.  Do not construct WMAData on a complete cache hit;
    # doing so defeats the purpose of pre-materializing the input.
    data = None
    if not all_input_caches_available:
        logging.info("***** Configing Data *****")
        with (timing.stage("data_module_setup") if timing else nullcontext()):
            data = instantiate_from_config(config.data)
            data.setup()
        print(">>> Dataset is successfully loaded ...")

    with (timing.stage("model_to_gpu") if timing else nullcontext()):
        transfer_mode = os.environ.get('UNIFOLM_MODEL_TO_GPU_NONBLOCKING', '0')
        if transfer_mode not in {'0', '1'}:
            raise ValueError('Invalid model GPU transfer mode')
        if transfer_mode == '1':
            model = model.to(device='cuda', non_blocking=True)
        else:
            model = model.cuda()
        print(f'MODEL_TO_GPU_NONBLOCKING enabled={transfer_mode} '
              'scope=original_timed_model_to_gpu dtype=preserved', flush=True)
        if timing and torch.cuda.is_available():
            torch.cuda.synchronize()
    device = get_device_from_parameters(model)
    with (timing.stage("inference_optimization_setup") if timing else nullcontext()):
        from unifolm_wma.utils.inference_runtime import configure_denoiser
        configure_denoiser(model, os.environ.get("UNIFOLM_DENOISER_MODE", "native"), args.dataset)
        from unifolm_wma.utils import crossstep_feature_runtime as _crossstep_runtime
        _crossstep_runtime.install(model, **{'input_start': 9, 'output_end': 3, 'start': 10, 'stop': 40})

    input_caches = {}
    if cache_paths:
        with (timing.stage("input_cache_load") if timing else nullcontext()):
            input_caches = {
                fs: load_input_cache(cache_path, args, fs, model)
                for fs, cache_path in cache_paths.items()
            }
        print(f">>> Loaded {len(input_caches)} preprocessed input cache(s).")

    # Run over data
    assert (args.height % 16 == 0) and (
        args.width % 16
        == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"

    # Get latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'>>> Generate {n_frames} frames under each generation ...')
    noise_shape = [args.bs, channels, n_frames, h, w]

    # A cache represents exactly one source CSV row.  The fixed benchmark has
    # one row per dataset; retaining the normal loop keeps general source runs
    # unchanged when no cache is requested.
    samples = [None] if all_input_caches_available else [
        df.iloc[idx] for idx in range(len(df))
    ]
    if args.write_input_cache and len(samples) != 1:
        raise ValueError(
            "--write_input_cache requires exactly one CSV row per dataset; "
            "otherwise cache names would be ambiguous.")

    # Start inference
    for raw_sample in samples:

        # If many, test various frequence control and world-model generation
        for fs in args.frame_stride:
            cache = input_caches.get(fs)
            if cache is not None:
                sample = cache["sample"]
                batch = cache["batch"]
                ori_state_dim = cache["ori_state_dim"]
                ori_action_dim = cache["ori_action_dim"]
                model_input_fs = cache["model_input_fs"]
            else:
                sample = raw_sample
                assert sample is not None
                init_frame_path = get_init_frame_path(args.prompt_dir, sample)
                ori_fps = float(sample['fps'])

                # Load transitions only when materializing a cache or running
                # from source files.  It is deliberately outside timed model
                # compute and absent on a cache hit.
                with (timing.stage("transition_h5_read") if timing else
                      nullcontext()):
                    import h5py
                    transition_path = get_transition_path(args.prompt_dir,
                                                          sample)
                    with h5py.File(transition_path, 'r') as h5f:
                        transition_dict = {}
                        for key in h5f.keys():
                            transition_dict[key] = torch.tensor(h5f[key][()])
                        for key in h5f.attrs.keys():
                            transition_dict[key] = h5f.attrs[key]

                start_idx = 0
                model_input_fs = ori_fps // fs
                with (timing.stage("initial_input_preprocess") if timing else
                      nullcontext()):
                    batch, ori_state_dim, ori_action_dim = prepare_init_input(
                        start_idx,
                        init_frame_path,
                        transition_dict,
                        fs,
                        data.test_datasets[args.dataset],
                        n_obs_steps=model.n_obs_steps_imagen)

            video_save_dir = args.savedir + f"/inference/sample_{sample['videoid']}"
            with (timing.stage("output_directory_setup") if timing else
                  nullcontext()):
                os.makedirs(video_save_dir, exist_ok=True)
                os.makedirs(video_save_dir + '/dm', exist_ok=True)
                os.makedirs(video_save_dir + '/wm', exist_ok=True)

            # For saving imagens in policy
            sample_save_dir = f'{video_save_dir}/dm/{fs}'
            os.makedirs(sample_save_dir, exist_ok=True)
            # For saving environmental changes in world-model
            sample_save_dir = f'{video_save_dir}/wm/{fs}'
            os.makedirs(sample_save_dir, exist_ok=True)
            # For collecting interaction videos
            wm_video = []
            # Initialize observation queues
            cond_obs_queues = {
                "observation.images.top":
                deque(maxlen=model.n_obs_steps_imagen),
                "observation.state": deque(maxlen=model.n_obs_steps_imagen),
                "action": deque(maxlen=args.video_length),
            }
            # Build the model's initial observation from a raw preprocessing
            # result or from its byte-for-byte cached equivalent.
            observation = {
                'observation.images.top':
                batch['observation.image'].permute(1, 0, 2,
                                                   3)[-1].unsqueeze(0),
                'observation.state':
                batch['observation.state'][-1].unsqueeze(0),
                'action':
                torch.zeros_like(batch['action'][-1]).unsqueeze(0)
            }
            observation = {
                key: observation[key].to(device, non_blocking=True)
                for key in observation
            }
            sample_full_video_file = (
                f"{video_save_dir}/../{sample['videoid']}_full_fs{fs}.mp4")
            # Update observation queues
            cond_obs_queues = populate_queues(cond_obs_queues, observation)

            if cache is not None:
                # Restore at the precise point at which a source run captures
                # the state: after model/data/input setup and before sampling.
                restore_random_state(cache["rng_state"])
            elif args.write_input_cache:
                assert args.input_cache_dir is not None
                cache_path = input_cache_path(args.input_cache_dir,
                                              args.dataset, fs)
                write_input_cache(cache_path, args, fs, model, sample, batch,
                                  ori_state_dim, ori_action_dim,
                                  model_input_fs)
                print(f">>> Wrote input cache: {cache_path}")

            if args.prepare_input_cache_only:
                continue

            # Multi-round interaction with the world-model
            for itr in tqdm(range(args.n_iter)):

                # Get observation
                observation = {
                    'observation.images.top':
                    torch.stack(list(
                        cond_obs_queues['observation.images.top']),
                                dim=1).permute(0, 2, 1, 3, 4),
                    'observation.state':
                    torch.stack(list(cond_obs_queues['observation.state']),
                                dim=1),
                    'action':
                    torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }

                # Use world-model in policy to generate action
                print(f'>>> Step {itr}: generating actions ...')
                with operator_profile("decision", itr), (
                        timing.stage("decision_generation", cuda=True)
                        if timing else nullcontext()):
                    pred_videos_0, pred_actions, _ = image_guided_synthesis_sim_mode(
                            model,
                            sample['instruction'],
                            observation,
                            noise_shape,
                            action_cond_step=args.exe_steps,
                            ddim_steps=args.ddim_steps,
                            ddim_eta=args.ddim_eta,
                            unconditional_guidance_scale=args.
                            unconditional_guidance_scale,
                            fs=model_input_fs,
                            timestep_spacing=args.timestep_spacing,
                            guidance_rescale=args.guidance_rescale,
                            sim_mode=False,
                            generation_stage="decision",
                            dataset=args.dataset)

                # Update future actions in the observation queues
                for idx in range(len(pred_actions[0])):
                    observation = {'action': pred_actions[0][idx:idx + 1]}
                    observation['action'][:, ori_action_dim:] = 0.0
                    cond_obs_queues = populate_queues(cond_obs_queues,
                                                      observation)

                # Collect data for interacting the world-model using the predicted actions
                observation = {
                    'observation.images.top':
                    torch.stack(list(
                        cond_obs_queues['observation.images.top']),
                                dim=1).permute(0, 2, 1, 3, 4),
                    'observation.state':
                    torch.stack(list(cond_obs_queues['observation.state']),
                                dim=1),
                    'action':
                    torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }

                # Interaction with the world-model
                print(f'>>> Step {itr}: interacting with world model ...')
                with operator_profile("simulation", itr), (
                        timing.stage("simulation_generation", cuda=True)
                        if timing else nullcontext()):
                    pred_videos_1, _, pred_states = image_guided_synthesis_sim_mode(
                        model,
                        "",
                        observation,
                        noise_shape,
                        action_cond_step=args.exe_steps,
                        ddim_steps=args.ddim_steps,
                        ddim_eta=args.ddim_eta,
                        unconditional_guidance_scale=args.
                        unconditional_guidance_scale,
                        fs=model_input_fs,
                        text_input=False,
                        timestep_spacing=args.timestep_spacing,
                        guidance_rescale=args.guidance_rescale,
                        generation_stage="simulation",
                        dataset=args.dataset)

                for idx in range(args.exe_steps):
                    observation = {
                        'observation.images.top':
                        pred_videos_1[0][:, idx:idx + 1].permute(1, 0, 2, 3),
                        'observation.state':
                        torch.zeros_like(pred_states[0][idx:idx + 1]) if
                        args.zero_pred_state else pred_states[0][idx:idx + 1],
                        'action':
                        torch.zeros_like(pred_actions[0][-1:])
                    }
                    observation['observation.state'][:, ori_state_dim:] = 0.0
                    cond_obs_queues = populate_queues(cond_obs_queues,
                                                      observation)

                with (timing.stage("iteration_result_write") if timing else
                      nullcontext()):
                    # Save the imagen videos for decision-making
                    sample_tag = f"{args.dataset}-vid{sample['videoid']}-dm-fs-{fs}/itr-{itr}"
                    if writer is not None:
                        log_to_tensorboard(writer,
                                           pred_videos_0,
                                           sample_tag,
                                           fps=args.save_fps)
                    # Save videos environment changes via world-model interaction
                    sample_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/itr-{itr}"
                    if writer is not None:
                        log_to_tensorboard(writer,
                                           pred_videos_1,
                                           sample_tag,
                                           fps=args.save_fps)

                    # Save the imagen videos for decision-making
                    sample_video_file = f'{video_save_dir}/dm/{fs}/itr-{itr}.mp4'
                    video_writer.write(pred_videos_0,
                                       sample_video_file,
                                       fps=args.save_fps)
                    # Save videos environment changes via world-model interaction
                    sample_video_file = f'{video_save_dir}/wm/{fs}/itr-{itr}.mp4'
                    pred_videos_1_cpu = video_writer.write_world(
                        pred_videos_1,
                        sample_video_file,
                        sample_full_video_file,
                        args.exe_steps,
                        fps=args.save_fps)

                print('>' * 24)
                # Collect the result of world-model interactions
                if not stream_full_video or writer is not None:
                    wm_video.append(
                        pred_videos_1_cpu[:, :, :args.exe_steps])

            with (timing.stage("full_video_write") if timing else
                  nullcontext()):
                sample_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/full"
                if writer is not None:
                    full_video = torch.cat(wm_video, dim=2)
                    log_to_tensorboard(writer,
                                       full_video,
                                       sample_tag,
                                       fps=args.save_fps)
                if not stream_full_video:
                    full_video = torch.cat(wm_video, dim=2)
                    video_writer.write(full_video,
                                       sample_full_video_file,
                                       fps=args.save_fps)

    with (timing.stage("video_writer_close") if timing else nullcontext()):
        video_writer.close()
    if os.environ.get("UNIFOLM_WORLD_DECODE_PROTECTED_TF32_CHUNK", "0") != "0":
        if os.environ.get("UNIFOLM_WORLD_DECODE_NONFEEDBACK_CACHE", "0") != "0":
            if os.environ.get("UNIFOLM_WORLD_DECODE_NONFEEDBACK_OVERLAP","0")!="0":
                from unifolm_wma.utils.vae_nonfeedback_overlap_runtime import get_report
                print("VAE_NONFEEDBACK_OVERLAP_CAPTURE "+json.dumps(get_report()),flush=True)
            else:
                from unifolm_wma.utils.vae_nonfeedback_cache_runtime import get_report
                print("VAE_NONFEEDBACK_CACHE_CAPTURE " + json.dumps(get_report()), flush=True)
        else:
            from unifolm_wma.utils.vae_protected_tf32_runtime import get_report
            print("VAE_PROTECTED_TF32_CAPTURE " + json.dumps(get_report()), flush=True)

    with (timing.stage("tensorboard_close") if timing else nullcontext()):
        if writer is not None:
            writer.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",
                        type=str,
                        default=None,
                        help="Path to save the results.")
    parser.add_argument(
        "--profile_path",
        type=str,
        default=None,
        help=("Optional JSON output for phase timings. CUDA-event timings are "
              "recorded for the two generation phases."))
    parser.add_argument(
        "--print_compute_time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Print total decision + world-model CUDA time at completion, "
              "excluding all loading and output saves (default: enabled)."))
    parser.add_argument("--ckpt_path",
                        type=str,
                        default=None,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config",
                        type=str,
                        help="Path to the model checkpoint.")
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Directory containing videos and corresponding prompts.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=("Optional WMAData root. Defaults to the value in --config; set "
              "this when evaluating a self-contained prompt directory."))
    parser.add_argument(
        "--input_cache_dir",
        type=str,
        default=None,
        help=("Directory of locally generated preprocessed-input/RNG snapshots. "
              "A hit bypasses source CSV/H5/PNG reads and WMAData setup."))
    parser.add_argument(
        "--require_input_cache",
        action="store_true",
        help=("Fail if an input cache is absent instead of falling back to "
              "raw sample preprocessing."))
    parser.add_argument(
        "--write_input_cache",
        action="store_true",
        help=("Materialize the source sample's preprocessed input and RNG "
              "checkpoint in --input_cache_dir."))
    parser.add_argument(
        "--prepare_input_cache_only",
        action="store_true",
        help=("Perform setup/cache materialization then exit before any model "
              "sampling or video writes. Requires --write_input_cache when "
              "the cache is absent."))
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="the name of dataset to test")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps. If non-positive, DDPM is used instead.")
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="Eta for DDIM sampling. Set to 0.0 for deterministic results.")
    parser.add_argument("--bs",
                        type=int,
                        default=1,
                        help="Batch size for inference. Must be 1.")
    parser.add_argument("--height",
                        type=int,
                        default=320,
                        help="Height of the generated images in pixels.")
    parser.add_argument("--width",
                        type=int,
                        default=512,
                        help="Width of the generated images in pixels.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        nargs='+',
        required=True,
        help=
        "frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)"
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=1.0,
        help="Scale for classifier-free guidance during sampling.")
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed for reproducibility.")
    parser.add_argument("--video_length",
                        type=int,
                        default=16,
                        help="Number of frames in the generated video.")
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="uniform",
        help=
        "Strategy for timestep scaling. See Table 2 in the paper: 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help=
        "Rescale factor for guidance as discussed in 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--perframe_ae",
        action='store_true',
        default=False,
        help=
        "Use per-frame autoencoder decoding to reduce GPU memory usage. Recommended for models with resolutions like 576x1024."
    )
    parser.add_argument(
        "--exe_steps",
        type=int,
        default=16,
        help="num of samples to execute",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=40,
        help="num of iteration to interact with the world model",
    )
    parser.add_argument("--zero_pred_state",
                        action='store_true',
                        default=False,
                        help="not using the predicted states as comparison")
    parser.add_argument("--save_fps",
                        type=int,
                        default=8,
                        help="fps for the saving video")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.write_input_cache and args.input_cache_dir is None:
        parser.error("--write_input_cache requires --input_cache_dir")
    if args.prepare_input_cache_only and not args.write_input_cache:
        parser.error("--prepare_input_cache_only requires --write_input_cache")
    # A compute-time summary needs CUDA events even when no JSON profile is
    # requested. Pass --no-print_compute_time to retain the old no-timing
    # behavior for a manual run.
    timing = TimingCollector(PROCESS_START_TIME) if (
        args.profile_path or args.print_compute_time) else None
    with (timing.stage("reproducibility_setup") if timing else nullcontext()):
        configure_reproducibility(args.seed)
        from unifolm_wma.utils.inference_runtime import configure_math_mode
        configure_math_mode(os.environ.get("UNIFOLM_MATH_MODE", "fp32"), args.dataset)
    run_inference(args, timing)
    if timing and args.print_compute_time:
        timing.print_compute_summary()
    if timing:
        from unifolm_wma.utils import crossstep_feature_runtime as _crossstep_runtime
        _crossstep_runtime.finish(args)
        timing.write(args.profile_path)
