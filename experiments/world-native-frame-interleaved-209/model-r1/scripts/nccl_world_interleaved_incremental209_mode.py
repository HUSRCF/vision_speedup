"""Keep adopted world routing enabled for both paired graph conditions."""
from contextlib import contextmanager
import nccl_spatial_both_temporal209_mode as parallel
import nccl_world_interleaved209_mode as residual
from nccl_world_interleaved209_mode import measure
active=None


class Controller:
    def __init__(self,model):
        self.parallel=parallel.install(model);self.residual=residual.install(model)
    def report(self):
        return dict(self.residual.report(),adopted_routing=self.parallel.report(),
            baseline_is_adopted_world_algorithm=True,paired_difference_is_only_world_ResBlock_sharding=True)


def install(model):
    global active
    assert active is None;active=Controller(model);return active


def uninstall():
    global active
    residual.uninstall();parallel.uninstall();active=None


@contextmanager
def mode(value):
    with parallel.mode(True),residual.mode(value):yield
