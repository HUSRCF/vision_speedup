"""Check all split values, graph refresh, storage and local allocation scope."""
import json
import os
from unittest.mock import patch
import torch
from torch.utils import deterministic
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils import half_pack_allocation as allocation


def exact(a,b):
    for left,right in zip(a,b):
        assert torch.equal(left.view(torch.int16),right.view(torch.int16))


@torch.no_grad()
def main():
    torch.manual_seed(877)
    torch.use_deterministic_algorithms(True)
    assert deterministic.fill_uninitialized_memory
    cases=0
    for shape in [(40960,320),(10240,640),(2560,1280),(640,1280),
            (4096,512),(1280,5120),(73,97),(1,1),(3,1031)]:
        value=torch.randn(*shape,device='cuda')*.1
        value.view(-1)[0]=0.
        if value.numel()>1:value.view(-1)[1]=-0.
        for scale in [1.,256.]:
            os.environ['UNIFOLM_HALF_PACK_NOFILL']='0'
            baseline=pack_half_weight(value,weight_scale=scale)
            os.environ['UNIFOLM_HALF_PACK_NOFILL']='1'
            allocation._debug_poison=float('nan')
            exact(pack_half_weight(value,weight_scale=scale),baseline)
            allocation._debug_poison=None
            parts=pack_half_weight(value,weight_scale=scale)
            exact(parts,baseline)
            ptrs=[p.data_ptr() for p in parts]
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):pack_half_weight(value,weight_scale=scale,out=parts)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                captured=pack_half_weight(value,weight_scale=scale,out=parts)
            value.add_(.0001)
            expected=pack_half_weight(value,weight_scale=scale)
            graph.replay()
            exact(captured,expected)
            assert [p.data_ptr() for p in captured]==ptrs
            assert deterministic.fill_uninitialized_memory
            native=torch.empty_like(value,dtype=torch.float16)
            assert torch.isnan(native).all()
            def filled(x):
                os.environ['UNIFOLM_HALF_PACK_NOFILL']='0'
                return pack_half_weight(x,weight_scale=scale)
            def candidate(x):
                os.environ['UNIFOLM_HALF_PACK_NOFILL']='1'
                return pack_half_weight(x,weight_scale=scale)
            before=graph_seconds(filled,value,5,3)
            seconds=graph_seconds(candidate,value,5,3)
            after=graph_seconds(filled,value,5,3)
            cases+=1
            print('HALF_PACK_NOFILL '+json.dumps(dict(shape=shape,scale=scale,
                before_s=before,after_s=after,candidate_s=seconds,
                conservative_speedup=min(before,after)/seconds)),flush=True)
    os.environ['UNIFOLM_HALF_PACK_NOFILL']='1'
    with patch.object(torch,'empty_like',side_effect=RuntimeError('Injected allocation failure')):
        try:allocation.empty_like(value)
        except RuntimeError:pass
        else:raise AssertionError('Allocation failure lost')
    assert deterministic.fill_uninitialized_memory
    print(f'HALF_PACK_NOFILL_CHECK passed=true cases={cases} '
          'all_parts_signed_zero_bitwise=true full_nan_poison=true '
          'dynamic_input_graph_refresh=true output_storage_reused=true '
          'native_fill_retained=true allocation_exception_restore=true '
          'full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
