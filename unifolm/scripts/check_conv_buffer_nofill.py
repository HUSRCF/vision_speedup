"""Check every allocated intermediate and result, plus actual shapes and graphs."""
import json
import os
from unittest.mock import patch
import torch
from torch.utils import deterministic
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils import conv_buffer_allocation as allocation
from unifolm_wma.utils.triton_implicit_temporal_conv import implicit_temporal_conv
from unifolm_wma.utils.triton_compensated_temporal_conv import compensated_temporal_conv
from unifolm_wma.utils.triton_packed_spatial_conv import packed_spatial_conv
from unifolm_wma.utils.triton_winograd_f2 import transform_filter_f2,winograd_f2_conv2d


def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    bits=torch.int16 if a.dtype==torch.float16 else torch.int32
    assert torch.equal(a.view(bits),b.view(bits))


@torch.no_grad()
def main():
    torch.manual_seed(919)
    torch.use_deterministic_algorithms(True)
    assert deterministic.fill_uninitialized_memory
    fixtures=[('implicit320','temporal',implicit_temporal_conv,(1,320,16,40,64),320,dict(group_m=8)),
        ('implicit640','temporal',implicit_temporal_conv,(1,640,16,20,32),640,dict(group_m=16)),
        ('packed1280medium','temporal',compensated_temporal_conv,(1,1280,16,10,16),1280,dict(block_m=64)),
        ('packed1280deep','temporal',compensated_temporal_conv,(1,1280,16,5,8),1280,dict(block_m=64)),
        ('spatial1280','spatial',packed_spatial_conv,(16,1280,5,8),1280,dict(splits=1)),
        ('spatial2560','spatial',packed_spatial_conv,(16,2560,5,8),1280,dict(splits=4)),
        ('f2medium1280','f2',winograd_f2_conv2d,(16,1280,10,16),1280,dict(block_m=64)),
        ('f2medium2560','f2',winograd_f2_conv2d,(16,2560,10,16),1280,dict(block_m=64))]
    for bias in [True,False]:
        for tag,kind,fn,options in [('implicit','temporal',implicit_temporal_conv,dict(group_m=8)),
            ('packed','temporal',compensated_temporal_conv,dict(fused_output=False)),
            ('spatial','spatial',packed_spatial_conv,dict(splits=4)),
            ('f2','f2',winograd_f2_conv2d,dict(block_m=32))]:
            shape=(2,37,5,3,5) if kind=='temporal' else (2,37,7,9)
            fixtures.append((tag+'-tails-'+str(bias),kind,fn,shape,70,dict(options,_bias=bias)))
    buffers_checked=0
    for name,kind,fn,shape,outputs,options in fixtures:
        options=dict(options);bias_on=options.pop('_bias',True)
        value=torch.randn(*shape,device='cuda')*.1
        weight_shape=(outputs,shape[1]*3) if kind=='temporal' else (outputs,shape[1],3,3)
        weight=torch.randn(*weight_shape,device='cuda')*.01
        parts=transform_filter_f2(weight) if kind=='f2' else pack_half_weight(weight.flatten(1))
        bias=torch.randn(outputs,device='cuda')*.03 if bias_on else None
        def run(x,mode):
            os.environ['UNIFOLM_CONV_BUFFER_NOFILL']=mode
            return fn(x,parts,bias,**options)
        allocations={}
        original=allocation.empty
        for mode in ['0',kind]:
            records=[]
            def tracked(*args,**kwargs):
                result=original(*args,**kwargs);records.append(result);return result
            allocation._debug_poison=float('nan') if mode==kind else None
            with patch.object(allocation,'empty',side_effect=tracked):
                # Imported aliases remain bound to original, so instrument each helper alias too.
                with patch(fn.__module__+'.empty_conv_buffer',side_effect=tracked):result=run(value,mode)
            allocations[mode]=(result,records)
        allocation._debug_poison=None
        exact(allocations['0'][0],allocations[kind][0])
        assert allocations['0'][1] and len(allocations['0'][1])==len(allocations[kind][1])
        for baseline,actual in zip(allocations['0'][1],allocations[kind][1]):
            assert torch.isfinite(actual).all(),(name,'Unwritten or nonfinite intermediate')
            exact(baseline,actual);buffers_checked+=1
        del allocations
        before=graph_seconds(lambda x:run(x,'0'),value,5,3)
        seconds=graph_seconds(lambda x:run(x,kind),value,5,3)
        after=graph_seconds(lambda x:run(x,'0'),value,5,3)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):run(value,kind)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=run(value,kind)
        value.add_(.0001);weight.mul_(.999)
        if kind=='f2':transform_filter_f2(weight,out=parts)
        else:pack_half_weight(weight.flatten(1),out=parts)
        if bias is not None:bias.add_(.0001)
        expected=run(value,'0');graph.replay();exact(captured,expected)
        assert deterministic.fill_uninitialized_memory and torch.isnan(torch.empty_like(value)).all()
        print('CONV_BUFFER_NOFILL '+json.dumps(dict(name=name,kind=kind,shape=shape,outputs=outputs,
            bias=bias_on,before_s=before,after_s=after,candidate_s=seconds,
            conservative_speedup=min(before,after)/seconds)),flush=True)
    os.environ['UNIFOLM_CONV_BUFFER_NOFILL']='all'
    with patch.object(torch,'empty',side_effect=RuntimeError('Injected allocation failure')):
        try:allocation.empty('temporal',(1,),device='cuda')
        except RuntimeError:pass
        else:raise AssertionError('Allocation error lost')
    assert deterministic.fill_uninitialized_memory
    print(f'CONV_BUFFER_NOFILL_CHECK passed=true cases={len(fixtures)} intermediates={buffers_checked} '
          'actual_temporal_spatial_f2_shapes=true every_intermediate_nan_poison_and_signed_zero_bits=true '
          'odd_shapes_bias_branches_splitk=true graph_input_master_bias_refresh=true '
          'native_fill_retained=true allocation_exception_restore=true full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
