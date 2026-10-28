"""Measure native ResBlock grids on top of the adopted parallel routing.

Both baseline and candidate use the existing spatial/temporal parallel mode.
Only the candidate enables the new ResBlock native spatial-grid partition.
This module is for paired diagnostics, not a production feature flag.
"""
from contextlib import contextmanager
import nccl_spatial_both_temporal209_mode as parallel
import nccl_resblock_native_grid209_mode as residual
from nccl_resblock_native_grid209_mode import measure

active=None


class Controller:
    def __init__(self,model):
        self.parallel=parallel.install(model)
        self.residual=residual.install(model)

    def report(self):
        a=self.residual.report()
        return dict(a, adopted_routing=self.parallel.report(),
            baseline_is_qualified_parallel_algorithm=True,
            paired_difference_is_only_new_ResBlock_parallelism=True)


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
    with parallel.mode(True),residual.mode(value):
        yield
