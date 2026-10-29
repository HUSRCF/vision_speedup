"""Partition frame-independent ResBlock work before its full temporal convolution.

Both modes retain the qualified original63 dual-GPU Transformer path. This
candidate changes cuDNN's spatial batch descriptor and adds one gather per
selected ResBlock; complete quality and timing qualification remain necessary.
"""
from contextlib import contextmanager
import torch
import torch.distributed as dist
from nonv2_all_blocks59_mode import install as original_install
from nonv2_all_blocks59_mode import uninstall as original_uninstall
from nonv2_all_blocks59_mode import mode as original_mode
from nonv2_ffn59_mode import candidate_cache

active = None


def spatial_then_temporal(value, embedding, rank, spatial, gather, temporal, batch_size):
    assert rank in (0, 1) and value.ndim == 4 and value.shape[0] == 16
    assert embedding.ndim == 2 and embedding.shape[0] in (1, 16)
    assert batch_size == 1
    local_slice = slice(8*rank, 8*(rank+1))
    emb = embedding if embedding.shape[0] == 1 else embedding[local_slice]
    local = spatial(value[local_slice], emb, batch_size=None)
    assert local.ndim == 4 and local.shape[0] == 8
    assert local.device == value.device and local.dtype == value.dtype
    output = torch.empty((16, *local.shape[1:]), device=local.device, dtype=local.dtype)
    gather(output, local.contiguous())
    # Preserve the original full16 temporal extent and original noncontiguous
    # B,C,T,H,W view, so temporal GroupNorm sees the complete statistics domain.
    if temporal is not None:
        output = temporal(output.unsqueeze(0).permute(0, 2, 1, 3, 4))
        output = output.permute(0, 2, 1, 3, 4).reshape(16, *output.shape[1:2], *output.shape[3:])
    return output


class Controller:
    def __init__(self, model):
        from unifolm_wma.modules.networks.wma_model import ResBlock
        self.inner = original_install(model)
        self.rank = self.inner.rank
        self.saved = self.inner.saved
        self.rows = self.inner.rows
        self.enabled = False
        self.hooks = []
        self.modules = []
        self.spatial_parameters = []
        self.captured = []
        for name, module in model.named_modules():
            if type(module) is not ResBlock or name not in {'output_blocks.7.0','output_blocks.8.0','output_blocks.9.0','output_blocks.10.0','output_blocks.11.0'}:
                continue
            assert not module.updown and not module.use_checkpoint
            before = module._forward
            def forward(value, embedding, batch_size=None, _before=before, _module=module, _name=name):
                if not self.enabled:
                    return _before(value, embedding, batch_size=batch_size)
                assert value.is_cuda and value.dtype == torch.float16
                assert not torch.is_grad_enabled() and not _module.training
                temporal = _module.temopral_conv if _module.use_temporal_conv else None
                with candidate_cache(self.rank):
                    result = spatial_then_temporal(value, embedding, self.rank, _before,
                        dist.all_gather_into_tensor, temporal, batch_size)
                if torch.cuda.is_current_stream_capturing():
                    self.captured.append(dict(module=_name, input_shape=list(value.shape),
                        output_shape=list(result.shape), local_frames=8,
                        added_collectives=1, temporal_full16=True,
                        full5D_temporal_GroupNorm_preserved=True,
                        local_send_bytes=result.numel()*result.element_size()//2))
                return result
            module._forward = forward
            self.hooks.append((module, before))
            self.modules.append(module)
            for child in (module.in_layers, module.emb_layers, module.out_layers, module.skip_connection):
                self.spatial_parameters.extend(child.parameters())
        assert len(self.modules) == 5

    def masters(self):
        parameters = [*self.inner.masters().values(), *self.spatial_parameters]
        return {id(p): p for p in parameters}

    def report(self):
        report = self.inner.report()
        report['resblock_frames'] = dict(modules=len(self.modules), captured=len(self.captured),
            rows=self.captured.copy(), original63_dual_Transformer_both_modes=True,
            full5D_temporal_GroupNorm_preserved=True,
            new_native_argument_or_address_inspection=False,
            spatial_batch_descriptor_changes=True, full_video_quality_required=True)
        return report


def install(model):
    global active
    assert active is None
    active = Controller(model)
    return active


def uninstall():
    global active
    if active is not None:
        for module, before in reversed(active.hooks):
            module._forward = before
    original_uninstall()
    active = None


@contextmanager
def mode(value):
    assert active is not None
    previous = active.enabled
    active.enabled = bool(value)
    try:
        with original_mode(True):
            yield
    finally:
        active.enabled = previous
