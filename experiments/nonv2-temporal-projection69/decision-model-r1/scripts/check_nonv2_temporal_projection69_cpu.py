"""Check both ranks' gather placement, dtype and current-master projection."""
import torch
from nonv2_temporal_projection69_mode import project_and_gather


@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    checked = 0
    for dtype in (torch.float16, torch.float32):
        for width in (8, 16):
            projection = torch.nn.Linear(width, width//2).to(dtype).eval()
            projection.weight.copy_(torch.arange(width*width//2).reshape(width//2, width)%8/32)
            projection.bias.copy_(torch.arange(width//2)/64)
            value = torch.arange(6*16*width).reshape(1, 6, 16, width).to(dtype)/1024
            original = projection(value)
            for mutation in (False, True):
                if mutation:
                    projection.weight.add_(.03125)
                expected = projection(value)
                for rank in (0, 1):
                    local = value[:, rank*3:(rank+1)*3].contiguous()
                    calls = []
                    def gather(full, shard):
                        assert not calls and shard.shape == (3, 16, width//2)
                        calls.append(1)
                        full.copy_(expected.squeeze(0))
                        full[rank*3:(rank+1)*3].copy_(shard)
                    actual = project_and_gather(local, rank, projection, gather)
                    assert torch.equal(actual, expected) and len(calls) == 1
                    assert actual.dtype == dtype and actual.shape == expected.shape
                    if mutation:
                        assert not torch.equal(actual, original)
                    checked += 1
    for shape in ((2, 3, 16, 8), (1, 3, 8, 8)):
        try:
            project_and_gather(torch.ones(shape), 0, lambda v:v, lambda a,b:None)
        except AssertionError:
            pass
        else:
            raise AssertionError('Unsupported batch/time shape accepted')
    assert not torch.cuda.is_initialized()
    print(f'TEMPORAL_PROJECTION69_CPU passed=true cases={checked} both_ranks=true current_masters_live=true one_collective=true no_GPU=true no_video_quality_inferred=true')


if __name__ == '__main__':
    main()
