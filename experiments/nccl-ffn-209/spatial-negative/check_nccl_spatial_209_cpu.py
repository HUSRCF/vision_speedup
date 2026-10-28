"""Check global frame masks, cache separation and hook restoration on CPU."""
import json
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.attention import SpatialTransformer,CrossAttention
import nccl_spatial_209_mode as candidate

def main():
    torch.manual_seed(17)
    model=torch.nn.ModuleList([SpatialTransformer(32,1,32,context_dim=32,use_linear=True,video_length=16,image_cross_attention=True) for _ in range(16)]).eval()
    masks=0;projections=0
    for rank in (0,1):
        attention=[m for m in model.modules() if isinstance(m,CrossAttention)]
        saved=[(m,m._get_attn_mask_aa,m._project_condition,m.prepare_condition_cache) for m in attention]
        with patch.object(candidate.dist,'get_world_size',return_value=2),patch.object(candidate.dist,'get_rank',return_value=rank):controller=candidate.install(model)
        try:
            for m,original_mask,original_projection,original_prepare in saved:
                m._condition_cache_enabled=True
                with torch.no_grad():
                    value=torch.randn(16,77,32)
                    with candidate.mode(False):baseline=m._project_condition('k',m.to_k,value)
                    cached=m._condition_projection_cache['k'];ptr=baseline.data_ptr();key=cached['key']
                    with candidate.mode(True):
                        actual=m._get_attn_mask_aa(8,1,256)
                        expected=original_mask(16,1,256)[rank*8:(rank+1)*8]
                        assert torch.equal(actual,expected)
                        first_allowed=int(torch.isfinite(actual[0,0]).sum());assert first_allowed==(rank*8+1)*16
                        masks+=1
                        piece=value[rank*8:(rank+1)*8]
                        result=m._project_condition('k',m.to_k,piece)
                        assert torch.equal(result,torch.nn.functional.linear(piece,m.to_k.weight))
                        assert cached is m._condition_projection_cache['k'] and cached['output'].data_ptr()==ptr and cached['key']==key
                        m.prepare_condition_cache(value)
                        assert cached is m._condition_projection_cache['k']
                        projections+=1
        finally:candidate.uninstall()
        assert all(m._get_attn_mask_aa==a and m._project_condition==b and m.prepare_condition_cache==c for m,a,b,c in saved)
    assert not torch.cuda.is_initialized()
    print('NCCL_SPATIAL_CPU_CHECK '+json.dumps(dict(passed=True,global_frame_masks=masks,uncached_candidate_projections=projections,baseline_cache_storage_and_keys_preserved=True,all_hooks_restored=True,no_GPU_initialized=True)))

if __name__=='__main__':main()
