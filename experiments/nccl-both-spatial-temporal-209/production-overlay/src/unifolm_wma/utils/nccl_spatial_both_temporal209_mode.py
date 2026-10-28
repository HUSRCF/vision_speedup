"""Whole spatial blocks plus temporal-only FFNs and all88 temporal convolutions."""
from contextlib import contextmanager
import torch
from . import nccl_spatial_decision209_mode as spatial
from . import nccl_temporal_209_mode as temporal
from . import nccl_ffn_209_mode as ffn
from .nccl_geglu_209_mode import measure

active=None

class Controller:
    def __init__(self,model):
        self.spatial=spatial.install(model)
        self.temporal=temporal.install(model)
        self.spatial.ffn_parent=ffn

    @property
    def captures(self):return self.spatial.captures+self.temporal.captures

    def report(self):
        a,b=self.spatial.report(),self.temporal.report()
        return dict(captured=a['captured']+b['captured'],spatial_captured=a['captured'],
            ffn_captured=b['captured'],temporal_captured=b['temporal_captured'],
            spatial=a,temporal=b,spatial_FFN_inner_allgather_disabled=True,two_GPUs=True)

def install(model):
    global active
    assert active is None;active=Controller(model);return active

def uninstall():
    global active
    temporal.uninstall();spatial.uninstall();active=None

@contextmanager
def mode(value):
    # Decision vendor GEMMs keep their original dimensions. The earlier
    # decision FFN sharder failed exactness, so only spatial JIT routing is
    # enabled there; all temporal/FFN hooks remain world-only.
    world=value and not torch.backends.cudnn.allow_tf32
    with spatial.mode(value),temporal.mode(world):yield
