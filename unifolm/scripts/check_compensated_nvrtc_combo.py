"""Compare the combined activation routes with the accepted compensated base."""
import torch
from unifolm_wma.modules.attention import GEGLU
from unifolm_wma.utils.triton_compensated_linear import install_compensated_linears
from unifolm_wma.utils.triton_compensated_geglu import install_compensated_geglus
from unifolm_wma.utils.nvrtc_geglu import install_nvrtc_geglus


@torch.no_grad()
def main():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    for channels in (320,640):
        reference=GEGLU(channels,channels*4).cuda().eval()
        candidate=GEGLU(channels,channels*4).cuda().eval()
        candidate.load_state_dict(reference.state_dict())
        keys=tuple(candidate.state_dict())
        for model in (reference,candidate):
            install_compensated_linears(model,narrow_output=True)
            install_compensated_geglus(model)
        assert install_nvrtc_geglus(candidate,fmad=True,vectorized=True)==1
        shapes=((2560,16,channels),(40,16,channels)) if channels==320 else ((40,16,channels),)
        for shape in shapes:
            value=torch.randn(shape,device='cuda')
            actual,expected=candidate(value),reference(value)
            assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):candidate(value)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):output=candidate(value)
            value.mul_(.93);graph.replay()
            expected=reference(value)
            assert torch.equal(output.view(torch.int32),expected.view(torch.int32))
            print('COMPENSATED_NVRTC_COMBO shape='+str(shape)+' passed=true',flush=True)
            del value,actual,expected,stream,graph,output
        reference.train();candidate.train()
        value=torch.randn(2,16,channels,device='cuda')
        torch.testing.assert_close(candidate(value),reference(value),rtol=0,atol=0)
        reference.eval();candidate.eval()
        with torch.enable_grad():
            value.requires_grad_(True)
            a,b=candidate(value),reference(value)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(torch.autograd.grad(a.sum(),value)[0],
                                      torch.autograd.grad(b.sum(),value)[0],rtol=0,atol=0)
        assert keys==tuple(candidate.state_dict())
        del reference,candidate,value,a,b
        torch.cuda.empty_cache()
    print('COMPENSATED_NVRTC_COMBO_CHECK passed=true full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
