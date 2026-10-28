"""Check complete16T sample ownership and mask slicing, without CUDA."""
import torch
from nonv2_temporal_block59_mode import temporal_inference
assert not torch.cuda.is_initialized();cases=0
for dtype in (torch.float16,torch.float32):
    for size in (40,160,640,2560):
        x=(torch.arange(size,dtype=torch.float32)[:,None,None]%13).expand(size,16,8).contiguous().to(dtype)
        full_mask=torch.arange(size)[:,None,None].expand(size,16,16).contiguous()
        for mask in(None,full_mask):
            expected=x*.5+.25
            for rank in(0,1):
                half=size//2;calls=[]
                def forward(local,context=None,mask=None):
                    assert context is None and local.shape==(half,16,8)
                    assert local.data_ptr()==x[rank*half].data_ptr()
                    if full_mask is not None and mask is not None:
                        assert torch.equal(mask,full_mask[rank*half:(rank+1)*half])
                    calls.append('forward');return local*.5+.25
                def gather(out,local):
                    assert out.shape==x.shape and torch.equal(local,expected[rank*half:(rank+1)*half])
                    calls.append('gather');out.copy_(expected)
                result=temporal_inference(x,rank,forward,gather,mask)
                assert torch.equal(result,expected)and calls==['forward','gather'];cases+=1
assert not torch.cuda.is_initialized()
print('NONV2_TEMPORAL_BLOCK59_CPU passed=true cases='+str(cases)+' full16T=true one_gather=true no_GPU=true')
