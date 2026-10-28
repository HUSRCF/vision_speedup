"""Require signed-zero exact old outputs for zero and nonzero filter residuals."""
import json
import os
import torch
from benchmark_temporal_conv2d import graph_seconds
from unifolm_wma.utils.triton_packed_spatial_conv import packed_spatial_conv,spatial_weight_parts
from unifolm_wma.utils.triton_winograd_f2 import transform_filter_f2,winograd_f2_conv2d
from unifolm_wma.utils.triton_zero_weight_low_spatial import residual_counts
from unifolm_wma.utils import conv_buffer_allocation as allocation


def exact(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert torch.equal(actual.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    torch.manual_seed(967)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    os.environ['UNIFOLM_CONV_BUFFER_NOFILL']='all'
    os.environ['UNIFOLM_HALF_PACK_NOFILL']='1'
    fixtures=[('packed',16,1280,1280,5,8,1),('packed',16,2560,1280,5,8,4),
              ('packed',2,37,70,7,9,1),('packed',2,37,70,7,9,4),
              ('f2',16,1280,1280,10,16,1),('f2',16,2560,1280,10,16,1),
              ('f2',16,1920,640,20,32,1),('f2',2,37,70,7,9,1)]
    for index,(kind,batch,inputs,outputs,h,w,splits) in enumerate(fixtures):
        for quantized in [True,False]:
            x=torch.randn(batch,inputs,h,w,device='cuda')*.3
            weight=torch.randn(outputs,inputs,3,3,device='cuda')*.02
            if quantized:weight=weight.half().float()
            bias=torch.randn(outputs,device='cuda')*.02 if index%2 else None
            transform=spatial_weight_parts if kind=='packed' else transform_filter_f2
            os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL']='all'
            parts=transform(weight)
            counts=residual_counts(parts,kind);assert counts is not None
            count_ptr=counts.data_ptr()
            zero_components=int((counts==0).sum())
            def run(value,mode):
                os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL']=mode
                if kind=='packed':return packed_spatial_conv(value,parts,bias,splits=splits)
                return winograd_f2_conv2d(value,parts,bias,block_m=32 if index==5 else 64)
            reference=run(x,'0')
            allocation._debug_poison=float('nan')
            actual=run(x,kind)
            allocation._debug_poison=None
            assert torch.isfinite(actual).all();exact(actual,reference)
            before=graph_seconds(lambda v:run(v,'0'),x,5,3)
            seconds=graph_seconds(lambda v:run(v,kind),x,5,3)
            after=graph_seconds(lambda v:run(v,'0'),x,5,3)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):run(x,kind)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):captured=run(x,kind)
            x.mul_(.997);weight.add_(.0001)
            if bias is not None:bias.add_(.0001)
            os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL']=kind
            transform(weight,out=parts)
            assert residual_counts(parts,kind).data_ptr()==count_ptr
            graph.replay();exact(captured,run(x,'0'))
            print('ZERO_WEIGHT_LOW_SPATIAL '+json.dumps(dict(index=index,kind=kind,
                shape=[batch,inputs,outputs,h,w],splits=splits,quantized=quantized,
                zero_filter_components=zero_components,total_filter_components=counts.numel(),
                before_s=before,candidate_s=seconds,after_s=after,
                conservative_speedup=min(before,after)/seconds,exact_bits=True,
                master_refresh_same_count_pointer=True,graph_input_master_bias_refresh=True)),flush=True)
            del x,weight,bias,parts,counts,actual,reference,captured,graph
            torch.cuda.empty_cache()
    # Missing counts during a cold capture must use the already warmed original.
    for kind in ['packed','f2']:
        x=torch.randn(2,37,7,9,device='cuda');w=torch.randn(70,37,3,3,device='cuda')*.01
        transform=spatial_weight_parts if kind=='packed' else transform_filter_f2
        os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL']='0';parts=transform(w)
        compute=lambda:packed_spatial_conv(x,parts) if kind=='packed' else winograd_f2_conv2d(x,parts)
        for _ in range(3):compute()
        expected=compute();torch.cuda.synchronize()
        os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_SPATIAL']=kind
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=compute()
        assert not hasattr(parts[0],'_spatial_filter_residual_counts')
        graph.replay();exact(captured,expected)
        assert residual_counts(parts,kind) is not None
        with torch.inference_mode():
            unknown=tuple(torch.empty_like(p) for p in parts)
            assert residual_counts(unknown,kind) is None
    print('ZERO_WEIGHT_LOW_SPATIAL_CHECK passed=true cases=16 signed_zero_bits=true '
          'actual_and_odd_shapes=true zero_and_general_filters=true nan_poison=true '
          'graph_input_master_bias_refresh=true count_pointer_stable=true '
          'cold_capture_fallback=true unversioned_parts_fallback=true full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
