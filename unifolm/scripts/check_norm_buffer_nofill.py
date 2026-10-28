"""Bitwise, poison and graph checks for normalization buffer prefill removal."""
import json
import os
import torch
from torch.utils import deterministic
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils import norm_buffer_allocation as allocation
from unifolm_wma.utils.triton_split_group_norm import split_group_norm_silu
from unifolm_wma.utils.triton_residual_layer_norm import fused_residual_layer_norm


def exact(actual, expected):
    if not isinstance(actual, tuple):
        actual, expected = (actual,), (expected,)
    for left, right in zip(actual, expected):
        assert left.shape == right.shape and left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert torch.equal(left.view(torch.int16 if left.dtype == torch.float16 else torch.int32),
                           right.view(torch.int16 if right.dtype == torch.float16 else torch.int32))


def check(function, value, parameters, kind, label):
    os.environ['UNIFOLM_NORM_BUFFER_NOFILL'] = '0'
    expected = function(value)
    for scope in ('group', 'residual', 'all'):
        os.environ['UNIFOLM_NORM_BUFFER_NOFILL'] = scope
        allocation._debug_poison = float('nan')
        exact(function(value), expected)
        assert deterministic.fill_uninitialized_memory
    allocation._debug_poison = None
    os.environ['UNIFOLM_NORM_BUFFER_NOFILL'] = '0'
    before = graph_seconds(function, value, 10, 3)
    os.environ['UNIFOLM_NORM_BUFFER_NOFILL'] = kind
    candidate = graph_seconds(function, value, 10, 3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = function(value)
    value.add_(.01)
    for p in parameters:
        p.add_(.0001)
    expected = function(value)
    graph.replay()
    exact(captured, expected)
    assert deterministic.fill_uninitialized_memory
    os.environ['UNIFOLM_NORM_BUFFER_NOFILL'] = '0'
    after = graph_seconds(function, value, 10, 3)
    print('NORM_BUFFER_NOFILL ' + json.dumps(dict(kind=kind, label=label,
        shape=list(value.shape), dtype=str(value.dtype), before_s=before,
        after_s=after, candidate_s=candidate,
        conservative_speedup=min(before, after)/candidate,
        poison_bitwise=True, graph_input_weight_refresh=True)), flush=True)


@torch.no_grad()
def main():
    torch.manual_seed(751)
    torch.use_deterministic_algorithms(True)
    deterministic.fill_uninitialized_memory = True
    os.environ['UNIFOLM_NORM_BUFFER_NOFILL'] = 'all'
    try:
        allocation._allocate('group', lambda: (_ for _ in ()).throw(RuntimeError('probe')))
    except RuntimeError as error:
        assert str(error) == 'probe'
    else:
        raise AssertionError('Allocation exception probe failed')
    assert deterministic.fill_uninitialized_memory
    for dtype in (torch.float32, torch.float16):
        for shape in [(16, 320, 40, 64), (16, 640, 20, 32),
                      (16, 1280, 10, 16), (16, 1280, 5, 8),
                      (1, 320, 16, 40, 64), (1, 640, 16, 20, 32),
                      (1, 1280, 16, 10, 16), (1, 1280, 16, 5, 8),
                      (2, 96, 3, 7, 11)]:
            value = torch.randn(shape, device='cuda', dtype=dtype) * .5
            norm = torch.nn.GroupNorm(32, shape[1]).cuda().to(dtype).eval()
            check(lambda x: split_group_norm_silu(x, norm), value,
                  tuple(norm.parameters()), 'group', 'contiguous_silu')
            if shape == (2, 96, 3, 7, 11):
                value = value.transpose(-1, -2)
                check(lambda x: split_group_norm_silu(x, norm, apply_silu=False),
                      value, tuple(norm.parameters()), 'group', 'strided_no_silu')
        for rows, channels in [(40960, 320), (10240, 640), (2560, 1280),
                               (40960, 512), (640, 1280), (73, 97), (0, 320)]:
            value = torch.randn(rows, channels, device='cuda', dtype=dtype) * .5
            right = torch.randn_like(value) * .3
            norm = torch.nn.LayerNorm(channels).cuda().to(dtype).eval()
            check(lambda x: fused_residual_layer_norm(x, right, norm), value,
                  (*norm.parameters(), right), 'residual', 'actual_and_tail')
    for bad in ('bad', '1'):
        os.environ['UNIFOLM_NORM_BUFFER_NOFILL'] = bad
        try:
            allocation.empty('group', (3,), device='cuda')
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid allocation scope accepted')
    os.environ['UNIFOLM_NORM_BUFFER_NOFILL'] = 'all'
    assert torch.isnan(torch.empty(17, device='cuda')).all()
    assert deterministic.fill_uninitialized_memory and torch.are_deterministic_algorithms_enabled()
    print('NORM_BUFFER_NOFILL_CHECK passed=true cases=34 all_scopes=true '
          'nan_poison_signed_zero_bitwise=true input_weight_graph_refresh=true '
          'native_unmodified_allocations_still_filled=true exception_restore=true '
          'deterministic_algorithms=true full_video_quality=unverified', flush=True)


if __name__ == '__main__':
    main()
