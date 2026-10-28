"""Adopted parallel routing plus qualified decision native ResBlock grids."""
from contextlib import contextmanager
import torch
from . import nccl_spatial_both_temporal209_mode as parallel
from . import nccl_resblock_native_grid209_mode as residual

active=None


class Controller:
    def __init__(self,model):
        self.parallel=parallel.install(model)
        self.residual=residual.install(model)

    @property
    def captures(self):
        # Preserve original runtime's decision16/world32 graph accounting.
        # The extra22 ResBlocks are checked separately in finish().
        return self.parallel.captures

    def report(self):
        return dict(self.parallel.report(),resblock_grid=self.residual.report())


def install(model):
    global active
    assert active is None
    active=Controller(model)
    return active


def uninstall():
    global active
    residual.uninstall()
    parallel.uninstall()
    active=None


@contextmanager
def mode(value):
    with parallel.mode(value),residual.mode(value and torch.backends.cudnn.allow_tf32):
        yield
