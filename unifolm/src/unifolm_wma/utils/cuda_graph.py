"""Replay the complete deterministic denoiser with fresh inputs every call.

Each signature owns a separate memory pool, and returned outputs are cloned so
the next replay cannot overwrite tensors retained by the DDIM sampler.
"""
import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


def _cached_normalize(module):
    constants = {}

    def forward(tensor):
        if not tensor.is_floating_point():
            raise TypeError("Normalize expects a floating-point tensor")
        if tensor.ndim < 3:
            raise ValueError("Normalize expects (..., C, H, W)")
        key = tensor.device, tensor.dtype
        if key not in constants:
            mean = torch.as_tensor(module.mean, dtype=tensor.dtype, device=tensor.device)
            std = torch.as_tensor(module.std, dtype=tensor.dtype, device=tensor.device)
            if (std == 0).any():
                raise ValueError("Normalize standard deviation must be nonzero")
            if mean.ndim == 1:
                mean = mean.view(-1, 1, 1)
            if std.ndim == 1:
                std = std.view(-1, 1, 1)
            constants[key] = mean, std
        mean, std = constants[key]
        result = tensor if module.inplace else tensor.clone()
        return result.sub_(mean).div_(std)

    return forward


def prepare_graph_constants(module):
    """Cache fixed torchvision normalization constants during eager warm-up."""
    from torchvision.transforms import Normalize
    count = 0
    for child in module.modules():
        if type(child) is Normalize:
            child.forward = _cached_normalize(child)
            count += 1
    return count


class DenoiserGraph:
    def __init__(self, original, max_graphs=2, condition_refresher=None,
                 skip_unchanged_inputs=False):
        self.original = original
        self.max_graphs = max_graphs
        self.entries = {}
        self.fallback_keys = set()
        self.condition_refresher = condition_refresher
        self.skip_unchanged_inputs = skip_unchanged_inputs

    @staticmethod
    def tensor_source_key(value):
        return (value.data_ptr(), value._version, tuple(value.shape),
                tuple(value.stride()), value.dtype, value.device)

    @staticmethod
    def condition_identity(args):
        if len(args) < 5:
            return None
        leaves, _ = tree_flatten(args[4])
        return tuple(("tensor", id(value)) if isinstance(value, torch.Tensor)
                     else (type(value).__name__, value) for value in leaves)

    @staticmethod
    def signature(leaves, spec):
        metadata = []
        for value in leaves:
            if isinstance(value, torch.Tensor):
                metadata.append(("tensor", tuple(value.shape), tuple(value.stride()),
                                 value.dtype, value.device))
            elif value is None or isinstance(value, (bool, int, float, str)):
                metadata.append((type(value).__name__, value))
            else:
                raise TypeError(f"Unsupported graph input: {type(value)}")
        return str(spec), tuple(metadata)

    @staticmethod
    def backend_math_key():
        return ("backend_math", bool(torch.backends.cudnn.allow_tf32),
                bool(torch.backends.cuda.matmul.allow_tf32))

    def __call__(self, *args, **kwargs):
        if torch.is_grad_enabled() or self.original.__self__.training:
            return self.original(*args, **kwargs)
        leaves, spec = tree_flatten((args, kwargs))
        tensors = [value for value in leaves if isinstance(value, torch.Tensor)]
        if not tensors or not all(value.is_cuda for value in tensors):
            return self.original(*args, **kwargs)
        key = self.signature(leaves, spec) + (self.backend_math_key(),)
        if key not in self.entries:
            if len(self.entries) >= self.max_graphs:
                if key not in self.fallback_keys:
                    print("CUDA_GRAPH_FALLBACK reason=signature_limit", flush=True)
                    self.fallback_keys.add(key)
                return self.original(*args, **kwargs)
            self.entries[key] = self.capture(args, kwargs, leaves, spec)
        else:
            entry = self.entries[key]
            copied_count = copied_bytes = skipped_count = skipped_bytes = 0
            for index, (static, current) in enumerate(
                    zip(entry["leaves"], leaves)):
                if isinstance(current, torch.Tensor):
                    source_key = self.tensor_source_key(current)
                    if (not self.skip_unchanged_inputs or
                            source_key != entry["source_keys"][index]):
                        static.copy_(current)
                        copied_count += 1
                        copied_bytes += current.numel() * current.element_size()
                        entry["source_keys"][index] = source_key
                        # Holding the previous source prevents allocator address
                        # reuse from masquerading as an unchanged tensor.
                        entry["sources"][index] = current
                    else:
                        skipped_count += 1
                        skipped_bytes += current.numel() * current.element_size()
            if self.skip_unchanged_inputs and entry["replays"] == 0:
                print("CUDA_GRAPH_INPUT_COPY "
                      f"copied={copied_count} copied_mib={copied_bytes / 2**20:.3f} "
                      f"skipped={skipped_count} "
                      f"skipped_mib={skipped_bytes / 2**20:.3f}", flush=True)
            entry["replays"] += 1
            condition_identity = self.condition_identity(args)
            if (self.condition_refresher is not None and
                    condition_identity != entry["condition_identity"]):
                static_args, static_kwargs = tree_unflatten(entry["leaves"], spec)
                self.condition_refresher(static_args, static_kwargs)
                entry["condition_identity"] = condition_identity
                print("CONDITION_CACHE_REFRESH", flush=True)
            if entry.get("numerical_sgemm") is not None:
                entry["numerical_sgemm"].replay()
            elif entry.get("native_weight_cache") is not None:
                entry["native_weight_cache"].replay()
            else:
                entry["graph"].replay()
        return tree_map(lambda value: value.clone(), self.entries[key]["outputs"])

    def capture(self, args, kwargs, leaves, spec):
        device = next(value.device for value in leaves if isinstance(value, torch.Tensor))
        static_leaves = [value.detach().clone() if isinstance(value, torch.Tensor)
                         else value for value in leaves]
        static_args, static_kwargs = tree_unflatten(static_leaves, spec)
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        from .native_graph_weight_cache import requested
        cache_weights = requested()
        from .native_graph_sgemm import requested as numerical_requested
        numerical_mode = numerical_requested()
        numerical_sgemm = None
        weight_cache = None
        if cache_weights or numerical_mode:
            graph.enable_debug_mode()
        # Warm-up and capture must neither advance the sampler's RNG nor
        # silently capture a stochastic forward pass.
        with torch.random.fork_rng(devices=[device.index]):
            cpu_rng = torch.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state(device)
            expected = self.original(*args, **kwargs)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.original(*static_args, **static_kwargs)
            current.wait_stream(stream)
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph, stream=stream):
                outputs = self.original(*static_args, **static_kwargs)
            from .decision_zero_low_runtime import report_capture as decision_zero_low_report_capture
            decision_zero_low_report_capture()
            from .world_zero_low_runtime import report_capture as world_zero_low_report_capture
            world_zero_low_report_capture()
            from .world_zero_low_stages_runtime import report_capture as protected_scores_report_capture
            protected_scores_report_capture()
            from .decision_zero_low_stages_runtime import report_capture as decision_stages_report_capture
            decision_stages_report_capture()
            from .decision_inline_zero_low_stages_runtime import report_capture as decision_inline_stages_report_capture
            decision_inline_stages_report_capture()
            from .serial_group_norm_runtime import report_capture as serial_norm_report_capture
            serial_norm_report_capture()
            from .world_two_product_tiles_runtime import report_capture as two_product_tiles_report_capture
            two_product_tiles_report_capture()
            from .norm_projection_pack_runtime import report_capture as norm_projection_pack_report_capture
            norm_projection_pack_report_capture()
            from .out_projection_residual_runtime import report_capture as out_projection_residual_report_capture
            out_projection_residual_report_capture()
            from .world_persistent_zero_low_runtime import report_capture as world_persistent_zero_low_report_capture
            world_persistent_zero_low_report_capture()
            if cache_weights:
                from .native_graph_weight_cache import NativeGraphWeights
                weight_cache = NativeGraphWeights(graph, self.original.__self__)
                weight_cache.replay()
            else:
                graph.replay()
            torch.cuda.synchronize(device)
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
                    gpu_rng, torch.cuda.get_rng_state(device)):
                raise RuntimeError("Graph warm-up/capture changed the denoiser RNG state")
            expected_leaves, expected_spec = tree_flatten(expected)
            output_leaves, output_spec = tree_flatten(outputs)
            if expected_spec != output_spec:
                raise RuntimeError("Graph output structure differs from eager execution")
            for actual, reference in zip(output_leaves, expected_leaves):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                if cache_weights and isinstance(actual, torch.Tensor) and actual.dtype == torch.float32:
                    assert torch.equal(actual.contiguous().view(torch.int32), reference.contiguous().view(torch.int32))
        if numerical_mode:
            assert weight_cache is None
            from .native_graph_sgemm import NativeSgemmModel
            numerical_sgemm = NativeSgemmModel(graph, self.original.__self__)
            numerical_sgemm.replay()
            errors = []
            for actual, reference in zip(output_leaves, expected_leaves):
                assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
                delta = actual.double() - reference.double()
                relative = float(delta.norm() / reference.double().norm().clamp_min(1e-30))
                maximum = float(delta.abs().max())
                assert relative <= .001 and maximum <= .001 * max(1., float(reference.abs().max()))
                errors.append(dict(relative_l2=relative,max_abs=maximum))
            import json
            print('NATIVE_SGEMM_FIRST_CAPTURE '+json.dumps(dict(outputs=errors,
                native_capture_bitwise=True,numerical_replay=True,full_video_PSNR_required=True)),flush=True)
        print(f"CUDA_GRAPH_CAPTURE index={len(self.entries)} "
              f"allocated_gib={torch.cuda.memory_allocated(device) / 2**30:.3f} "
              f"cudnn_tf32={torch.backends.cudnn.allow_tf32} "
              f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
              + ("native_capture_match=bitwise replay_math=numerical" if numerical_mode else "eager_match=bitwise"), flush=True)
        source_keys = [self.tensor_source_key(value)
                       if isinstance(value, torch.Tensor) else None
                       for value in leaves]
        sources = [value if isinstance(value, torch.Tensor) else None
                   for value in leaves]
        return {"graph": graph, "native_weight_cache": weight_cache, "numerical_sgemm": numerical_sgemm, "leaves": static_leaves, "outputs": outputs,
                "source_keys": source_keys, "sources": sources,
                "replays": 0,
                "condition_identity": self.condition_identity(args)}
