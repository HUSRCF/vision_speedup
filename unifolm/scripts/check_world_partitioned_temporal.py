"""Real Conv3d wrappers, FP32 masters, cache reuse, fallback, and cold capture."""
import json
import os
from unittest.mock import patch
import torch
from unifolm_wma.utils.triton_compensated_temporal_conv import make_compensated_temporal_forward
from unifolm_wma.utils.world_partitioned_temporal import make_forward,install_world_partitioned_temporal
import unifolm_wma.utils.world_partitioned_temporal as host


def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
    assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


@torch.no_grad()
def main():
    os.environ['UNIFOLM_IMPLICIT_TEMPORAL_CONV']='1'
    os.environ['UNIFOLM_CONV_BUFFER_NOFILL']='all'
    torch.manual_seed(772)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for channels,h,w in [(320,40,64),(640,20,32)]:
        for bias in [False,True]:
            root=torch.nn.Module()
            root.test=torch.nn.Conv3d(channels,channels,(3,1,1),padding=(1,0,0),bias=bias)
            root.other=torch.nn.Conv3d(channels,channels,1)
            root.deep=torch.nn.Conv3d(1280,1280,(3,1,1),padding=(1,0,0))
            root.eval().cuda()
            root.deep.forward=make_compensated_temporal_forward(root.deep)
            module=root.test
            module.weight.copy_(module.weight.half().float())
            module.forward=make_compensated_temporal_forward(module)
            original,other,deep=module.forward,root.other.forward,root.deep.forward
            state={k:(p.data_ptr(),p.dtype,p.shape) for k,p in root.state_dict().items()}
            assert install_world_partitioned_temporal(root)==1
            assert root.other.forward==other and root.deep.forward==deep
            for scope in ['all','packed','invalid']:
                try:install_world_partitioned_temporal(root,scope);raise AssertionError('Badscopeaccepted')
                except ValueError:pass
            try:install_world_partitioned_temporal(root);raise AssertionError('Duplicateaccepted')
            except ValueError:pass
            value=torch.randn(1,channels,16,h,w,device='cuda')*.7
            pointer_rows=[]
            call=host.partitioned_temporal_conv
            def observed(x,parts,count,b=None,**kwargs):
                pointer_rows.append(tuple(p.data_ptr() for p in (*parts,count)))
                return call(x,parts,count,b,**kwargs)
            with patch.object(host,'partitioned_temporal_conv',observed):
                exact(module(value),original(value))
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):module(value)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):captured=module(value)
                for quantized in [False,True,False]:
                    value.add_(.003);module.weight.add_(.00013)
                    if quantized:module.weight.copy_(module.weight.half().float())
                    if module.bias is not None:module.bias.add_(.0002)
                    eager=module(value);graph.replay()
                    exact(captured,eager);exact(captured,original(value))
                assert len(pointer_rows)>6 and len(set(pointer_rows))==1
            for fallback in ['decision','training','grad','autocast','noncontiguous','other_shape','matmul_tf32']:
                module.train(fallback=='training')
                torch.backends.cudnn.allow_tf32=fallback=='decision'
                torch.backends.cuda.matmul.allow_tf32=fallback=='matmul_tf32'
                x=(value.transpose(-1,-2).contiguous().transpose(-1,-2) if fallback=='noncontiguous' else
                   value[:,:,:3].contiguous() if fallback=='other_shape' else value)
                with torch.set_grad_enabled(fallback=='grad'):
                    with torch.autocast('cuda',enabled=fallback=='autocast',dtype=torch.float16):
                        with patch.object(host,'partitioned_temporal_conv',side_effect=AssertionError('Fallbackenteredcandidate')):
                            actual,expected=module(x),original(x)
                            bits=torch.int32 if actual.dtype==torch.float32 else torch.int16
                            assert torch.equal(actual.contiguous().view(bits),expected.contiguous().view(bits))
                module.eval();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            module.forward=original;module.forward=make_forward(module)
            expected=original(value);stream.wait_stream(torch.cuda.current_stream())
            cold=torch.cuda.CUDAGraph()
            with torch.cuda.graph(cold,stream=stream):cold_result=module(value)
            cold.replay();exact(cold_result,expected)
            assert state=={k:(p.data_ptr(),p.dtype,p.shape) for k,p in root.state_dict().items()}
            print('PARTITIONED_TEMPORAL_RUNTIME '+json.dumps(dict(channels=channels,bias=bias,
                signedzero=True,mutable_graph=True,same_cache_slots=True,zero_nonzero_zero=True,
                cold_capture=True,FP32_masters=True,unrelated_and_C1280_excluded=True,
                invalid_scope_rejected=True,duplicate_rejected=True,native_fallbacks=7)),flush=True)
            del root,module,value,graph,cold,captured,eager,expected,cold_result
            torch.cuda.empty_cache()
    print('PARTITIONED_TEMPORAL_RUNTIME_CHECK passed=true fixtures=4',flush=True)


if __name__=='__main__':main()
