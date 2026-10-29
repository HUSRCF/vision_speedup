"""Frame ownership, shared/per-frame embedding, and temporal boundary checks."""
import torch
from unifolm_wma.utils.nonv2_resblock_frames72_mode import spatial_then_temporal


def check():
    for embedding_rows in (1, 16):
        value = torch.arange(16*2*4*6, dtype=torch.float32).reshape(16, 2, 4, 6).transpose(2, 3)
        embedding = torch.arange(embedding_rows*2, dtype=torch.float32).reshape(embedding_rows, 2)
        def spatial(x, emb, batch_size):
            assert batch_size is None
            return x*.25+emb[:, :, None, None]
        expected_spatial = spatial(value, embedding, None)
        def temporal(x):
            assert x.shape == (1, 2, 16, 6, 4)
            return x + x.roll(1, dims=2)*.5
        expected = temporal(expected_spatial.unsqueeze(0).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).squeeze(0)
        for rank in (0, 1):
            def gather(output, local):
                assert torch.equal(local, expected_spatial[rank*8:(rank+1)*8])
                output.copy_(expected_spatial)
            actual = spatial_then_temporal(value, embedding, rank, spatial, gather, temporal, 1)
            assert torch.equal(actual, expected)
            actual = spatial_then_temporal(value, embedding, rank, spatial, gather, None, 1)
            assert torch.equal(actual, expected_spatial)
    assert not torch.cuda.is_initialized()
    print('RESBLOCK_FRAMES72_CPU passed=true both_rank_frame_ownership=true embedding1_and16=true full16_temporal_boundary=true no_GPU=true')


if __name__ == '__main__':
    check()
