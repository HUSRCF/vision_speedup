"""Factory membership and live inference guards for three isolated world roles."""
import json
from unittest.mock import patch
import torch
import torch.nn.functional as F
from unifolm_wma.utils.triton_compensated_linear import make_forward as precise_forward
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_two_product_linear import two_product_linear
from unifolm_wma.utils.world_two_product_roles import make_forward, install_world_two_product_roles
from unifolm_wma.utils.decision_primary_narrow import install_decision_primary_narrow


def identical(left, right):
    assert torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(661)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    for scope, count in [('qk320', 2), ('output640', 3), ('ffn640', 1)]:
        union = torch.nn.Module()
        union.attn = torch.nn.Module()
        for name in ('to_q', 'to_k', 'to_v'):
            setattr(union.attn, name, torch.nn.Linear(320, 320, bias=False))
        union.attn.to_out = torch.nn.Sequential(torch.nn.Linear(640, 640))
        union.block = torch.nn.Module()
        union.block.proj_in = torch.nn.Linear(640, 640)
        union.block.proj_out = torch.nn.Linear(640, 640)
        union.block.ff = torch.nn.Module()
        union.block.ff.net = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(),
                                                torch.nn.Linear(2560, 640))
        union.unrelated = torch.nn.Linear(640, 640)
        for name, module in union.named_modules():
            if isinstance(module, torch.nn.Linear) and name != 'unrelated':
                module._tf32x3_installed = True
                if module.in_features == 320:
                    module._decision_primary_narrow_installed = True
        originals = {name: module.forward for name, module in union.named_modules()
                     if isinstance(module, torch.nn.Linear)}
        assert install_world_two_product_roles(union, scope, 'input') == count
        for name, module in union.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            selected = (name.endswith(('.to_q', '.to_k')) if scope == 'qk320' else
                        name.endswith(('.to_out.0', '.proj_in', '.proj_out'))
                        if scope == 'output640' else name.endswith('.ff.net.2'))
            assert bool(getattr(module, '_world_two_product_roles_installed', False)) == selected
            if not selected:
                assert module.forward == originals[name]
        try:
            install_world_two_product_roles(union, scope, 'input')
            raise AssertionError('Duplicate role installation accepted')
        except ValueError:
            pass
    for scope in ('qk320', 'output640', 'ffn640'):
        inputs = 320 if scope == 'qk320' else 640 if scope == 'output640' else 2560
        outputs = 320 if scope == 'qk320' else 640
        rows = 40960 if scope == 'qk320' else 10240
        modes = ('input', 'weight') if scope == 'qk320' else ('input',)
        for protect in modes:
            for bias_enabled in (False, True):
                root = torch.nn.Module()
                if scope == 'qk320':
                    root.attn = torch.nn.Module()
                    root.attn.to_q = torch.nn.Linear(inputs, outputs, bias=bias_enabled)
                    module = root.attn.to_q
                elif scope == 'output640':
                    root.block = torch.nn.Module()
                    root.block.proj_in = torch.nn.Linear(inputs, outputs, bias=bias_enabled)
                    module = root.block.proj_in
                else:
                    root.block = torch.nn.Module()
                    root.block.ff = torch.nn.Module()
                    root.block.ff.net = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(),
                        torch.nn.Linear(inputs, outputs, bias=bias_enabled))
                    module = root.block.ff.net[2]
                root.eval().cuda()
                module.forward = precise_forward(module, min_matmul_macs=0,
                    backend='scaled-fp16', cache_half_weights=True, group_m=32,
                    block_k=64, pack_inputs=True)
                module._tf32x3_installed = True
                if scope == 'qk320':
                    assert install_decision_primary_narrow(root, 'all') == 1
                original = module.forward
                keys = tuple(root.state_dict())
                pointer = module.weight.data_ptr()
                assert install_world_two_product_roles(root, scope, protect) == 1
                value = torch.randn(rows, inputs, device='cuda') * .7
                torch.backends.cudnn.allow_tf32 = True
                with patch('unifolm_wma.utils.world_two_product_roles.two_product_linear',
                           side_effect=AssertionError('Decision entered world role experiment')):
                    identical(module(value), original(value))
                torch.backends.cudnn.allow_tf32 = False
                packed = []
                def record_pack(weight, *, out=None):
                    parts = pack_half_weight(weight, out=out)
                    packed.append(tuple(p.data_ptr() for p in parts))
                    return parts
                with patch('unifolm_wma.utils.world_two_product_roles.pack_half_weight',
                           side_effect=record_pack):
                    result = module(value)
                    parts = pack_half_weight(module.weight)
                    identical(result, two_product_linear(value, parts, module.bias, protect=protect))
                    assert len(packed) == 1
                    sample = [0, rows // 2, rows - 1]
                    exact = F.linear(value[sample].double(), module.weight.double(),
                                     None if module.bias is None else module.bias.double())
                    error = float((result[sample].double() - exact).norm() / exact.norm())
                    assert error < 6e-4 and torch.isfinite(result).all()
                    module(value)
                    assert len(packed) == 1
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            module(value)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = module(value)
                    value.add_(.01)
                    module.weight.add_(.00001)
                    if module.bias is not None:
                        module.bias.mul_(.95)
                    eager = module(value)
                    assert len(packed) == 2 and packed[0] == packed[1]
                    graph.replay()
                    identical(captured, eager)
                for fallback in ('training', 'grad', 'autocast', 'noncontiguous', 'shape', 'matmul_tf32'):
                    module.train(fallback == 'training')
                    torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
                    x = (value.t().contiguous().t() if fallback == 'noncontiguous'
                         else value[:35] if fallback == 'shape' else value)
                    with torch.set_grad_enabled(fallback == 'grad'):
                        with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                            with patch('unifolm_wma.utils.world_two_product_roles.two_product_linear',
                                       side_effect=AssertionError('Fallback entered world experiment')):
                                left, right = module(x), original(x)
                                assert left.dtype == right.dtype and torch.equal(left, right)
                    module.eval()
                    torch.backends.cuda.matmul.allow_tf32 = False
                module.forward = original
                module.forward = make_forward(module, protect=protect, rows=rows)
                expected = original(value)
                stream.wait_stream(torch.cuda.current_stream())
                cold = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cold, stream=stream):
                    cold_result = module(value)
                cold.replay()
                identical(cold_result, expected)
                assert keys == tuple(root.state_dict()) and module.weight.data_ptr() == pointer
                assert all(p.dtype == torch.float32 for p in root.parameters())
                print('WORLD_TWO_PRODUCT_ROLE_RUNTIME ' + json.dumps(dict(scope=scope,
                    protect=protect, bias=bias_enabled, shape=[rows, outputs, inputs],
                    fp64_relative=error, factory_scope_exclusion=True, decision_previous_bits=True,
                    cache_version_refresh_storage_reuse=True, graph_input_master_bias_refresh_bits=True,
                    native_fallbacks=6, cold_capture_fallback=True,
                    master_state_and_storage_preserved=True)), flush=True)
                del graph, cold, root, module, value, result, parts, captured, eager, cold_result
                torch.cuda.empty_cache()
    print('WORLD_TWO_PRODUCT_ROLE_RUNTIME_CHECK passed=true configs=8 scopes=3 '
          'full_video_quality=unverified official_score=false', flush=True)


if __name__ == '__main__':
    main()
