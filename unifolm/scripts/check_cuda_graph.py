"""Check graph input updates, scalar branches, RNG and output lifetimes."""
import importlib.util

import torch
from torchvision.transforms import Normalize

from unifolm_wma.utils.cuda_graph import DenoiserGraph, prepare_graph_constants
from unifolm_wma.utils.diffusion import timestep_embedding


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(16, 16)
        self.normalize = Normalize([0.5], [0.2])

    def forward(self, x, timestep, condition):
        normalized = self.normalize(x[None, None])[0, 0]
        value = self.projection(normalized) + condition["values"][0] + timestep_embedding(timestep, 16)
        if condition["reverse"]:
            value = -value
        return value, value.sum(dim=1), value.mean(dim=0)


class ConditionCachedModel(torch.nn.Module):
    """Minimal fixed-storage cache used to test graph refresh boundaries."""
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)
        self.cached = None
        self.key = None

    def forward(self, x, unused_action, unused_state, timestep, condition):
        source = condition["fixed"]
        self.prepare(source)
        return x + self.cached + timestep

    def prepare(self, source):
        key = source.data_ptr(), source._version
        if key != self.key:
            current = self.projection(source)
            if self.cached is None:
                self.cached = current
            else:
                self.cached.copy_(current)
            self.key = key


@torch.no_grad()
def main():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location("official_diffusion",
        "/share/problems/unifolm-world-model-action/src/unifolm_wma/utils/diffusion.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    timesteps = torch.tensor([999, 960, 20], device="cuda")
    for dimension in (320, 128, 65):
        torch.testing.assert_close(timestep_embedding(timesteps, dimension),
                                   official.timestep_embedding(timesteps, dimension),
                                   rtol=0, atol=0)
    model = Model().cuda().eval()
    normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image = torch.randn(2, 3, 32, 48, device="cuda").transpose(2, 3)
    reference = normalizer(image)
    assert prepare_graph_constants(normalizer) == 1
    torch.testing.assert_close(normalizer(image), reference, rtol=0, atol=0)
    assert prepare_graph_constants(model) == 1
    graph = DenoiserGraph(model.forward)
    x = torch.empty(16, 8, device="cuda").transpose(0, 1)
    timestep = torch.ones(1, device="cuda")
    bias = torch.zeros(16, device="cuda")
    held = []
    for index in range(6):
        x.normal_()
        timestep.fill_(index)
        bias.fill_(index / 7)
        condition = {"values": [bias], "reverse": bool(index % 2)}
        expected = model(x, timestep, condition)
        rng = torch.cuda.get_rng_state()
        actual = graph(x, timestep, condition)
        assert torch.equal(rng, torch.cuda.get_rng_state())
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        held.append((actual, expected))
    for actual, expected in held:
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert len(graph.entries) == 2
    cache_model = ConditionCachedModel().cuda().eval()
    refreshed = DenoiserGraph(
        cache_model.forward,
        condition_refresher=lambda values, options: cache_model.prepare(
            values[4]["fixed"]))
    retained = []
    for generation in range(3):
        fixed = torch.randn(2, 8, device="cuda")
        condition = {"fixed": fixed}
        for step in range(3):
            current = torch.randn(2, 8, device="cuda")
            timestep = torch.full((2, 1), step, device="cuda")
            expected = current + cache_model.projection(fixed) + timestep
            actual = refreshed(current, current, current, timestep, condition)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            retained.append((actual, expected))
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(refreshed.entries) == 1
    print("PASS: graph inputs, branches, RNG and retained outputs", flush=True)


if __name__ == "__main__":
    main()
