"""Decision ResBlock frame parallelism with original N16 Conv2d descriptors.

Each rank remaps its eight independent frames to the first half of an N16
tensor. Only validated native spatial grids are reduced. Original transformed
weight caches, kernel arguments, split-K, embedding projections and full16
temporal convolutions remain in place. One gather joins each spatial residual.
"""
import ast
import inspect
import json
import textwrap
from contextlib import contextmanager
import torch
import torch.distributed as dist
from audit_native_frame_grid209 import reduced_grid

enabled = False
active = None


def measure(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    exact = torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))
    record = dict(relative_l2=float((actual-expected).norm()/expected.norm().clamp_min(1e-30)),
        max_abs=float((actual-expected).abs().max()), signedzero_bits=exact,
        gross_passed=exact, full_video_PSNR_required=True)
    assert exact, record
    return record


def patched_forward(function, controller):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    original = ast.dump(tree, include_attributes=False)
    embeddings = [n for n in node.body if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == 'emb_out']
    assert len(embeddings) == 1
    emb = embeddings[0]
    value = emb.value
    emb.value = ast.Call(func=ast.Name(id='_parallel_embedding_rows', ctx=ast.Load()),
                         args=[value], keywords=[])
    sums = [n for n in node.body if isinstance(n, ast.Assign)
            and ast.unparse(n) == 'h = self.skip_connection(x) + h']
    assert len(sums) == 1
    gather = ast.parse('h = _parallel_spatial_gather(h)').body[0]
    node.body.insert(node.body.index(sums[0])+1, gather)
    prefix = ast.parse('x = _parallel_frame_rows(x)').body[0]
    node.body.insert(0, prefix)
    namespace = dict(function.__globals__, _parallel_frame_rows=controller.frame_rows,
        _parallel_embedding_rows=controller.embedding_rows,
        _parallel_spatial_gather=controller.gather)
    exec(compile(ast.fix_missing_locations(tree), '<resblock-native-frame-grid>', 'exec'), namespace)
    node.body.remove(prefix)
    node.body.remove(gather)
    emb.value = value
    assert ast.dump(tree, include_attributes=False) == original
    return namespace[function.__name__]


class Controller:
    def __init__(self, model):
        from unifolm_wma.modules.networks.wma_model import ResBlock
        from unifolm_wma.utils.native_graph_weight_cache import NativeGraphWeights
        assert dist.get_world_size() == 2
        self.rank = dist.get_rank()
        self.saved, self.hooks, self.captures, self.native_reports = [], [], [], []
        self.current = None
        self.geometry = {}
        function = patched_forward(ResBlock._forward, self)
        for name, module in model.named_modules():
            if type(module) is not ResBlock:
                continue
            old = module._forward
            assert old.__func__ is ResBlock._forward
            def forward(value, emb, batch_size=None, _module=module, _old=old, _name=name):
                if not enabled or not torch.backends.cudnn.allow_tf32:
                    return _old(value, emb, batch_size=batch_size)
                assert not torch.is_grad_enabled() and not _module.training and not torch.is_autocast_enabled()
                assert not torch.backends.cuda.matmul.allow_tf32 and self.current is None
                assert value.ndim == 4 and value.shape[0] == 16 and value.dtype == torch.float32
                assert batch_size in (None, 1)
                self.current = dict(module=_name, input_shape=list(value.shape),
                    input_stride=list(value.stride()), frame_start=self.rank*8,
                    frames=8, logical_frames=16, gathered=False, embedding_rows=None)
                try:
                    output = function(_module, value, emb, batch_size=batch_size)
                    assert self.current['gathered'] and output.shape[0] == 16
                    if torch.cuda.is_current_stream_capturing():
                        self.captures.append(dict(self.current))
                    return output
                finally:
                    self.current = None
            module._forward = forward
            self.saved.append((module, old))
            for suffix, conv in module.named_modules():
                if not isinstance(conv, torch.nn.Conv2d) or conv.kernel_size != (3, 3):
                    continue
                weight_name = name+'.'+suffix+'.weight'
                def observe(conv, args, _name=weight_name):
                    if self.current is None or not torch.cuda.is_current_stream_capturing():
                        return
                    value = args[0]
                    assert value.ndim == 4 and value.shape[0] == 16 and value.dtype == torch.float32
                    assert value.shape[1] == conv.in_channels and conv.stride == (1, 1)
                    record = dict(co=conv.out_channels, ci=conv.in_channels,
                        height=value.shape[2], width=value.shape[3],
                        input_stride=list(value.stride()), full_frames=16)
                    assert _name not in self.geometry or self.geometry[_name] == record
                    self.geometry[_name] = record
                self.hooks.append(conv.register_forward_pre_hook(observe))
        assert self.saved
        self.owner_class = NativeGraphWeights
        self.old_init = NativeGraphWeights.__init__
        def initialize(owner, graph, model):
            self.old_init(owner, graph, model)
            if not enabled:
                return
            # All original strict138+43+8 gates finish before this hook runs.
            assert self.current is None and len(self.geometry) == 2*len(self.saved)
            chosen = []
            for name, cache in owner.caches:
                if name not in self.geometry:
                    continue
                shape = self.geometry[name]
                assert tuple(cache.weight.shape) == (shape['co'], shape['ci'], 3, 3)
                params = cache.cached_consumer.params
                original = cache.original_consumer.params
                before = tuple(bytes(b) for b in cache.cached_consumer.buffers)
                grid = [params.grid_x, params.grid_y, params.grid_z]
                assert grid == [original.grid_x, original.grid_y, original.grid_z]
                new_grid, axis = reduced_grid(dict(name=cache.proof['consumer'], grid=grid),
                    shape['co'], shape['height'], shape['width'])
                params.grid_x, params.grid_y, params.grid_z = new_grid
                assert tuple(bytes(b) for b in cache.cached_consumer.buffers) == before
                assert [original.grid_x, original.grid_y, original.grid_z] == grid
                assert (params.func,params.block_x,params.block_y,params.block_z,params.shared) == (
                    original.func,original.block_x,original.block_y,original.block_z,original.shared)
                cache.set_cached(True)
                chosen.append(dict(weight=name, **shape, original_grid=grid, reduced_grid=new_grid,
                    axis=axis, consumer=cache.proof['consumer'], cached_filter_pointer_preserved=True,
                    all_cached_argument_bytes_unchanged=True, original_close_grid_preserved=True))
            assert len(chosen) == len(self.geometry), (len(chosen), len(self.geometry))
            record = dict(selected=len(chosen), total_original_cache_owners=len(owner.caches),
                original_native_report=dict(owner.report), rows=chosen,
                strict_original_cache_gates_completed=True, split_K_and_memsets_unchanged=True)
            self.native_reports.append(record)
            print('RESBLOCK_NATIVE_GRID '+json.dumps(record), flush=True)
        self.initialize = initialize
        NativeGraphWeights.__init__ = initialize

    def frame_rows(self, value):
        assert self.current is not None and value.shape[0] == 16
        output = torch.empty_strided(value.shape, value.stride(), device=value.device, dtype=value.dtype)
        output.zero_()
        output[:8].copy_(value[self.rank*8:(self.rank+1)*8])
        assert output.stride() == value.stride()
        return output

    def embedding_rows(self, value):
        assert self.current is not None and value.shape[0] in (1, 16)
        self.current['embedding_rows'] = value.shape[0]
        return value if value.shape[0] == 1 else self.frame_rows(value)

    def gather(self, value):
        assert self.current is not None and value.ndim == 4 and value.shape[0] == 16
        assert not self.current['gathered']
        output = torch.empty(value.shape, device=value.device, dtype=value.dtype)
        dist.all_gather_into_tensor(output, value[:8].contiguous())
        stride = [0]*4
        size = 1
        for dim in sorted(range(4), key=lambda i: value.stride()[i]):
            stride[dim] = size
            size *= value.shape[dim]
        if tuple(stride) != output.stride():
            arranged = torch.empty_strided(output.shape, tuple(stride),
                                           device=value.device, dtype=value.dtype)
            arranged.copy_(output)
            output = arranged
        self.current.update(gathered=True, spatial_output_shape=list(output.shape),
            spatial_output_stride=list(output.stride()), one_collective_before_original_temporal=True,
            full_embedding_projection_preserved=True)
        return output

    def report(self):
        assert self.current is None
        return dict(captured=len(self.captures), rows=self.captures.copy(),
            ResBlocks=len(self.saved), original_ResBlocks=len(self.saved), excluded_blocks=[],
            native_cache_coverage=self.native_reports.copy(), original_full_N16_descriptors=True,
            original_full16_temporal_and_heads=True, embedding_projection_full_shape=True,
            decision_only=True, full_video_PSNR_unverified=True, two_GPUs=True)


def install(model):
    global active
    assert active is None
    active = Controller(model)
    return active


def uninstall():
    global active, enabled
    if active is not None:
        assert active.current is None
        for module, old in reversed(active.saved):
            module._forward = old
        for hook in active.hooks:
            hook.remove()
        assert active.owner_class.__init__ == active.initialize
        active.owner_class.__init__ = active.old_init
    active = None
    enabled = False


@contextmanager
def mode(value):
    global enabled
    prior = enabled
    enabled = value
    try:
        yield
    finally:
        assert active is None or active.current is None
        enabled = prior
