"""Compare old untouched kernels against zero-residual GEGLU on real shapes."""
import json
import os
from unittest.mock import patch
import torch
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_world_narrow_geglu import primary_narrow_geglu
from unifolm_wma.utils.triton_branch_protected_geglu import branch_protected_geglu
from unifolm_wma.utils.triton_prepacked_geglu import prepacked_geglu
from unifolm_wma.utils import triton_zero_weight_low_geglu as zero
from unifolm_wma.utils import geglu_output_allocation as allocation


def exact(a,b):
    if not isinstance(a,tuple):a,b=(a,),(b,)
    for left,right in zip(a,b):
        assert left.shape==right.shape and left.dtype==right.dtype
        assert torch.equal(left.contiguous().view(torch.int32),right.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(991);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ.update(UNIFOLM_GEGLU_OUTPUT_NOFILL='1',UNIFOLM_HALF_PACK_NOFILL='1')
    fixtures=[('narrow',40960,320,1280,'gate',64,64),('wide',10240,640,2560,'gate',64,64),
        ('prepacked',2560,1280,5120,'compensated',64,32),('prepacked',640,1280,5120,'compensated',64,32),
        ('narrow',73,320,59,'compensated',64,32),('narrow',73,320,59,'value',64,64),
        ('narrow',73,320,59,'both',64,32),('prepacked',73,320,59,'compensated',32,32)]
    for index,(kind,rows,inner,channels,mode,bm,bn) in enumerate(fixtures):
        for quantized in [True,False]:
            value=torch.randn(rows,inner,device='cuda')*.3
            weight=torch.randn(channels*2,inner,device='cuda')*.02
            if quantized:weight=weight.half().float()
            bias=torch.randn(channels*2,device='cuda')*.02 if index%2 else None
            parts=pack_half_weight(weight)
            def compute(x,debug=False):
                if kind=='narrow':return primary_narrow_geglu(x,parts,bias,mode=mode,block_n=bn,debug_branches=debug)
                if kind=='wide':return branch_protected_geglu(x,parts,bias,mode=mode,block_n=bn,group_m=32,debug_branches=debug)
                return prepacked_geglu(x,parts,bias,block_m=bm,block_n=bn,group_m=32)
            def run(x,flag,debug=False):
                os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU']=flag
                return compute(x,debug)
            reference=run(value,'0',True)
            allocation._debug_poison=float('nan');actual=run(value,'all',True)
            allocation._debug_poison=None
            exact(actual,reference)
            os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU']='all'
            count=zero.residual_count(parts);assert count is not None
            count_ptr=count.data_ptr();assert bool(count==0)==quantized
            before=graph_seconds(lambda x:run(x,'0'),value,5,3)
            seconds=graph_seconds(lambda x:run(x,'all'),value,5,3)
            after=graph_seconds(lambda x:run(x,'0'),value,5,3)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):run(value,'all')
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):captured=run(value,'all',True)
            value.add_(.0001);weight.add_(.00013)
            if bias is not None:bias.add_(.0002)
            pack_half_weight(weight,out=parts)
            os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU']='all'
            zero.residual_count(parts,refresh=True)
            assert zero.residual_count(parts).data_ptr()==count_ptr and int(count)>0
            graph.replay();exact(captured,run(value,'0',True))
            weight.copy_(weight.half().float());pack_half_weight(weight,out=parts)
            os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU']='all';zero.residual_count(parts,refresh=True)
            assert int(count)==0 and zero.residual_count(parts).data_ptr()==count_ptr
            graph.replay();exact(captured,run(value,'0',True))
            torch.backends.cudnn.allow_tf32=True
            with patch.object(zero,'_zero_geglu',side_effect=AssertionError('Decision stage entered world kernel')):
                exact(run(value,'all'),run(value,'0'))
            torch.backends.cudnn.allow_tf32=False
            print('ZERO_WEIGHT_LOW_GEGLU '+json.dumps(dict(index=index,kind=kind,
                shape=[rows,inner,channels],mode=mode,block_m=bm,block_n=bn,
                group_m=16 if kind=='narrow' else 32,quantized=quantized,
                before_s=before,candidate_s=seconds,after_s=after,
                conservative_speedup=min(before,after)/seconds,exact_output_and_debug_bits=True,
                graph_changed_input_master_bias=True,count_pointer_stable_and_both_patterns=True)),flush=True)
            del value,weight,bias,parts,count,actual,reference,captured,graph
            torch.cuda.empty_cache()
    # Cold capture and unversioned tensors must keep the old kernels.
    value=torch.randn(73,320,device='cuda');weight=torch.randn(118,320,device='cuda')*.02
    parts=pack_half_weight(weight);os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU']='0'
    compute=lambda:primary_narrow_geglu(value,parts,mode='gate',block_n=32)
    for _ in range(3):compute()
    expected=compute();torch.cuda.synchronize()
    os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU']='all';graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=compute()
    assert not hasattr(parts[0],'_geglu_zero_weight_low_count')
    graph.replay();exact(captured,expected)
    with torch.inference_mode():
        unknown=tuple(torch.empty_like(p) for p in parts)
        assert zero.residual_count(unknown) is None
    print('ZERO_WEIGHT_LOW_GEGLU_CHECK passed=true cases=16 real_and_odd_shapes=true '
          'old_output_and_debug_signedzero_bits=true poison=true zero_and_general_weights=true '
          'original_primary_flags_BK_and_group_preserved=true graph_input_master_bias_refresh=true '
          'count_same_pointer_zero_nonzero_switch=true decision_fallback=true '
          'cold_capture_and_unversioned_fallback=true full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
