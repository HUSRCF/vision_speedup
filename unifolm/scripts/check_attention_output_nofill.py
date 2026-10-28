"""Check complete stores, exact bits and graph replay of scoped allocations."""
import json
import os
from unittest.mock import patch
import torch
from torch.utils import deterministic
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils import attention_buffer_allocation as allocation
from unifolm_wma.utils.triton_compensated_attention import compensated_attention
from unifolm_wma.utils.triton_temporal_attention_fp32 import fused_temporal_attention_fp32


def exact(actual, reference):
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert torch.equal(actual.contiguous().view(torch.int32), reference.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(941)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert deterministic.fill_uninitialized_memory
    fixtures = [('temporal', (b, 16, h*64), True) for b, h in
                [(2560,8),(2560,5),(640,10),(160,20),(40,20)]]
    fixtures += [('temporal', (3,16,320), False)]
    fixtures += [('spatial', (b,m,h,64), n) for b,m,n,h in
                 [(16,2560,2560,5),(16,640,640,10),(16,160,160,20),
                  (1,73,97,3),(2,1,3,1)]]
    for index, (kind, shape, option) in enumerate(fixtures):
        query = torch.randn(shape, device='cuda')*.2
        key_shape = shape if kind == 'temporal' else (shape[0],option,shape[2],64)
        key = torch.randn(key_shape, device='cuda')*.2
        value = torch.randn_like(key)*.3
        def compute(q):
            if kind == 'temporal':
                return fused_temporal_attention_fp32(q,key,value,shape[-1]//64,.125)
            return compensated_attention(q,key,value,block_m=64,block_n=64)
        def run(q, mode):
            os.environ['UNIFOLM_ATTENTION_OUTPUT_NOFILL'] = mode
            return compute(q)
        reference = run(query, '0')
        allocation._debug_poison = float('nan')
        actual = run(query, kind)
        allocation._debug_poison = None
        assert torch.isfinite(actual).all()
        exact(actual, reference)
        # A scope must not change other allocations, including native/debug ones.
        assert deterministic.fill_uninitialized_memory and torch.isnan(torch.empty_like(query)).all()
        before = graph_seconds(lambda q: run(q,'0'),query,5,3)
        candidate = graph_seconds(lambda q: run(q,kind),query,5,3)
        after = graph_seconds(lambda q: run(q,'0'),query,5,3)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): run(query,kind)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): captured = run(query,kind)
        query.add_(.0001);key.mul_(.999);value.add_(.0002)
        expected = run(query,'0')
        graph.replay();exact(captured,expected)
        if kind == 'temporal':
            os.environ['UNIFOLM_ATTENTION_OUTPUT_NOFILL'] = 'all'
            debug = fused_temporal_attention_fp32(query,key,value,shape[-1]//64,.125,
                       return_intermediates=True)
            exact(debug[0],expected)
            assert all(torch.isfinite(t).all() for t in debug)
        print('ATTENTION_OUTPUT_NOFILL '+json.dumps(dict(index=index,kind=kind,shape=shape,
            option=option,before_s=before,candidate_s=candidate,after_s=after,
            conservative_speedup=min(before,after)/candidate)),flush=True)
        del query,key,value,actual,reference,captured,expected,graph
        torch.cuda.empty_cache()
    os.environ['UNIFOLM_ATTENTION_OUTPUT_NOFILL'] = 'all'
    for previous in (True,False):
        deterministic.fill_uninitialized_memory = previous
        with patch.object(torch,'empty_like',side_effect=RuntimeError('Injected allocation failure')):
            try: allocation.empty_like('temporal',torch.zeros(1))
            except RuntimeError: pass
            else: raise AssertionError('Allocation error lost')
        assert deterministic.fill_uninitialized_memory == previous
    deterministic.fill_uninitialized_memory = True
    print('ATTENTION_OUTPUT_NOFILL_CHECK passed=true cases=11 poison_complete_stores=true '
          'signed_zero_bits=true graph_changed_qkv=true debug_path=true '
          'global_flag_restore_success_and_exception=true full_video_quality=unverified',flush=True)


if __name__ == '__main__': main()
