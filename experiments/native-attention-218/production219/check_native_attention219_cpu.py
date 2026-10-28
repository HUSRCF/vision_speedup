"""Check global frame/mask routing at the native boundary with no GPU."""
import json
from unittest.mock import patch
import torch
from unifolm_wma.utils.nccl_native_attention219_mode import local_native


def reference(q,k,v,attn_bias=None,p=0.,scale=None,op=None,output_dtype=None):
    assert p==0.
    scores=torch.einsum('bmhd,bnhd->bhmn',q,k)*(q.shape[-1]**-.5 if scale is None else scale)
    if attn_bias is not None:scores=scores+attn_bias
    return torch.einsum('bhmn,bnhd->bmhd',scores.softmax(-1),v).to(output_dtype or q.dtype)


@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized();torch.manual_seed(1870)
    checks=[]
    for keys in (2,16,77,256):
        q=torch.randn(16,7,3,8);k=torch.randn(16,keys,3,8);v=torch.randn(16,keys,3,8)
        for mask_kind in ('none','global_prefix','broadcast'):
            mask=None
            if mask_kind!='none':
                # Each frame has its own original global valid prefix.
                prefix=(torch.arange(16)+1)*max(1,keys//16)
                mask=torch.zeros(16,3,7,keys)
                mask.masked_fill_(torch.arange(keys)[None,None,None,:]>=prefix[:,None,None,None],float('-inf'))
                if mask_kind=='broadcast':mask=mask[:1]
            expected=reference(q,k,v,attn_bias=mask)
            rng=torch.get_rng_state().clone()
            for rank in (0,1):
                calls=[]
                def parent(a,b,c,**kw):
                    assert a.shape[0]==b.shape[0]==c.shape[0]==8
                    if mask is not None:
                        target=mask[rank*8:(rank+1)*8] if mask.shape[0]==16 else mask
                        assert torch.equal(kw['attn_bias'],target)
                    calls.append(1);return reference(a,b,c,**kw)
                result,record=local_native(parent,q,k,v,mask,rank=rank)
                actual=result[rank*8:(rank+1)*8]
                torch.testing.assert_close(actual,expected[rank*8:(rank+1)*8],rtol=0,atol=0)
                assert result.shape==expected.shape and result.dtype==expected.dtype and calls==[1]
                assert record['extra_collectives']==0 and record['local_batch']==8
                assert torch.equal(rng,torch.get_rng_state())
                checks.append(dict(keys=keys,mask=mask_kind,rank=rank,passed=True))
    assert not torch.cuda.is_initialized()
    print('NATIVE_ATTENTION218_CPU '+json.dumps(dict(passed=True,cases=len(checks),rows=checks,
        no_GPU_initialized=True,original_global_mask_slices=True,full_output_shape=True,
        parent_native_batch8=True,no_extra_collectives=True,RNG_unchanged=True,
        actual_CUDA_xformers_numerics_and_dispatch_unverified=True)),flush=True)


if __name__=='__main__':main()
