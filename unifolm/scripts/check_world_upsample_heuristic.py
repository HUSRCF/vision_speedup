"""Check global flag restoration on exceptions and strict named factory scope."""
import json
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.networks.wma_model import Upsample
from unifolm_wma.utils.world_upsample_heuristic import make_forward, install_world_upsample_heuristic


class Sentinel(Exception):
    pass


previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32)
fixtures = 0
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        for channels, spatial in [(1280,(20,32)),(640,(40,64))]:
            for bias in [False, True]:
                module = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=bias).cuda().eval()
                value = torch.empty((16,channels,*spatial),device='cuda')
                seen = []
                def raise_original(value):
                    seen.append(torch.backends.cudnn.benchmark)
                    raise Sentinel
                module.forward = raise_original
                wrapped = make_forward(module, channels)
                for tf32 in [False, True]:
                    torch.backends.cudnn.allow_tf32 = tf32
                    try:
                        wrapped(value)
                        raise AssertionError('Original exception must propagate')
                    except Sentinel:
                        pass
                    assert torch.backends.cudnn.benchmark is True
                    assert torch.backends.cudnn.allow_tf32 == tf32
                    assert torch.backends.cuda.matmul.allow_tf32 is False
                assert seen == [False,True], seen
                fixtures += 1
                torch.backends.cudnn.allow_tf32 = False
                del wrapped, value, module
        model = torch.nn.Module()
        model.output_blocks = torch.nn.ModuleList([torch.nn.Sequential() for _ in range(9)])
        for index, channels in [(2,1280),(5,1280),(8,640)]:
            model.output_blocks[index] = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(3)],
                                                           Upsample(channels, True))
        selected = install_world_upsample_heuristic(model)
        assert set(selected) == {'output_blocks.5.3.conv','output_blocks.8.3.conv'}
        assert not getattr(model.output_blocks[2][3].conv,'_world_upsample_heuristic_installed',False)
        try:
            install_world_upsample_heuristic(model)
            raise AssertionError('Duplicate installation must fail')
        except AssertionError as error:
            assert str(error) == 'output_blocks.5.3'
    assert fixtures == 4
    print('WORLD_UPSAMPLE_HEURISTIC_RUNTIME_CHECK '+json.dumps(dict(passed=True,exception_fixtures=4,
        decision_fallback_fixtures=4,global_flags_restored=True,factory_selected=selected,
        smaller_upsample_untouched=True,duplicate_rejected=True,no_official_score=True)),flush=True)
finally:
    torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous
