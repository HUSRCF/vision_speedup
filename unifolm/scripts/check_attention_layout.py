"""Check complete attention layers against the untouched official source."""
import importlib.util
from pathlib import Path

import torch

from unifolm_wma.modules.attention import CrossAttention


@torch.no_grad()
def main():
    spec = importlib.util.spec_from_file_location(
        "official_attention", Path("/share/problems/unifolm-world-model-action") /
        "src/unifolm_wma/modules/attention.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    comparisons = 0
    for context_dim, context_lengths in ((320, (None,)), (1024, (93, 95, 351))):
        settings = dict(query_dim=320, context_dim=context_dim, heads=5,
                        dim_head=64, image_cross_attention=True, video_length=16,
                        agent_state_context_len=2, agent_action_context_len=256)
        reference = official.CrossAttention(**settings).cuda().eval()
        modules = {}
        for layout in ("packed", "bmhk"):
            module = CrossAttention(**settings).cuda().eval()
            module.load_state_dict(reference.state_dict(), strict=True)
            module._inference_attention_layout = layout
            module._condition_cache_enabled = True
            modules[layout] = module
        for queries in (33, 160, 2560):
            for context_length in context_lengths:
                context = (None if context_length is None else
                           torch.randn(16, context_length, context_dim,
                                       device="cuda"))
                for _ in range(2):
                    # Second query exercises cached conditions with new data.
                    value = torch.randn(16, queries, 320, device="cuda")
                    expected = reference(value, context)
                    for layout, module in modules.items():
                        actual = module(value, context)
                        torch.testing.assert_close(actual, expected, rtol=0,
                                                   atol=0)
                        comparisons += 1
            print(f"ATTENTION_LAYOUT_LAYER_CHECK context_dim={context_dim} "
                  f"queries={queries} comparisons={comparisons} "
                  "bitwise_equal=true", flush=True)
    print(f"ATTENTION_LAYOUT_LAYER_CHECK passed=true comparisons={comparisons}",
          flush=True)


if __name__ == "__main__":
    main()
