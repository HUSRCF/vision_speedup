"""Move the existing temporal gather after Linear projection; no extra collective.

Use ordinary Torch modules/tensors. The complete 5D GroupNorm and original
projection/inner-block arithmetic remain live. Only the output Linear's batch
extent changes, requiring independent numerical and full-video qualification.
"""
from contextlib import contextmanager
import torch
import torch.distributed as dist
from unifolm_wma.modules.attention import TemporalTransformer
from nonv2_all_blocks59_mode import install as install_original
from nonv2_all_blocks59_mode import uninstall as uninstall_original
from nonv2_all_blocks59_mode import mode as original_mode
import nonv2_temporal_block59_mode as temporal

active = None
parent_inference = temporal.temporal_inference


def project_and_gather(value, rank, project, gather):
    assert rank in (0, 1) and value.ndim == 4 and value.shape[0] == 1
    assert value.shape[2] == 16 and value.is_contiguous()
    local = project(value)
    assert local.shape[:3] == value.shape[:3]
    assert local.dtype == value.dtype and local.device == value.device
    local = local.squeeze(0).contiguous()
    full = torch.empty((2*local.shape[0], *local.shape[1:]),
                       dtype=local.dtype, device=local.device)
    gather(full, local)
    return full.unsqueeze(0)


def inference(value, rank, forward, gather, mask=None):
    if active is None or active.scope is None:
        return parent_inference(value, rank, forward, gather, mask)
    # Keep the qualified slicing, mask and original block call unchanged.
    # Its unused full allocation is deliberately retained; measure total cost.
    retained = []
    def keep_local(output, local):
        assert not retained
        retained.append(local)
    parent_inference(value, rank, forward, keep_local, mask)
    assert len(retained) == 1
    return retained[0]


class Controller:
    def __init__(self, model):
        self.inner = install_original(model)
        self.rank = self.inner.rank
        self.saved = self.inner.saved
        self.rows = self.inner.rows
        self.scope = None
        self.enabled = False
        self.hooks = []
        self.projections = []
        self.deferred_rows = []
        for name, parent in model.named_modules():
            if type(parent) is not TemporalTransformer or not parent.use_linear:
                continue
            assert parent.only_self_att and not parent.causal_attention
            assert len(parent.transformer_blocks) == 1
            assert isinstance(parent.proj_out, torch.nn.Linear)
            before = parent.forward
            def forward(value, *args, _before=before, _name=name, **kwargs):
                if not self.enabled:
                    return _before(value, *args, **kwargs)
                assert self.scope is None
                assert value.ndim == 5 and value.shape[0] == 1 and value.shape[2] == 16
                assert value.is_cuda and value.dtype == torch.float16
                assert not torch.is_grad_enabled()
                self.scope = _name
                before_count = len(self.rows)
                try:
                    result = _before(value, *args, **kwargs)
                    if torch.cuda.is_current_stream_capturing():
                        assert len(self.rows) == before_count+1
                        row = self.rows[before_count]
                        assert row['scope'] == 'temporal_inner_block'
                        row['local_output_MiB'] = result.numel()*result.element_size()/2**21
                        row['output_projection_before_existing_gather'] = True
                    return result
                finally:
                    self.scope = None
            parent.forward = forward
            self.hooks.append((parent, before))
            before = parent.proj_out.forward
            def project(value, _before=before, _name=name):
                if not self.enabled:
                    return _before(value)
                assert self.scope == _name
                result = project_and_gather(value, self.rank, _before,
                                            dist.all_gather_into_tensor)
                if torch.cuda.is_current_stream_capturing():
                    self.deferred_rows.append(dict(module=_name,
                        local_projection_shape=list(value.shape),
                        full_projection_shape=list(result.shape),
                        existing_collective_moved=True, additional_collectives=0))
                return result
            parent.proj_out.forward = project
            self.hooks.append((parent.proj_out, before))
            self.projections.append(parent.proj_out)
        assert len(self.projections) == 16

    def masters(self):
        parameters = list(self.inner.masters().values())
        parameters += [p for module in self.projections for p in module.parameters()]
        return {id(p): p for p in parameters}

    def report(self):
        result = self.inner.report()
        result['full5D_GroupNorm_and_boundary_projections_outside_shard'] = False
        result['full5D_GroupNorm_outside_shard'] = True
        result['temporal_input_projection_outside_shard'] = True
        result['temporal_output_projection'] = dict(
            modules=16, captured=len(self.deferred_rows),
            rows=self.deferred_rows.copy(), full5D_GroupNorm_unchanged=True,
            init_Conv1d_projection_unchanged=True, additional_collectives=0,
            baseline_is_original63=True, full_video_quality_required=True)
        return result


def install(model):
    global active
    assert active is None and temporal.temporal_inference is parent_inference
    active = Controller(model)
    temporal.temporal_inference = inference
    return active


def uninstall():
    global active
    assert active is not None and active.scope is None
    for module, before in reversed(active.hooks):
        module.forward = before
    temporal.temporal_inference = parent_inference
    uninstall_original()
    active = None


@contextmanager
def mode(value):
    assert active is not None
    previous = active.enabled
    active.enabled = bool(value)
    try:
        # Both sides already use the qualified original63 dual-GPU path.
        with original_mode(True):
            yield
    finally:
        active.enabled = previous
