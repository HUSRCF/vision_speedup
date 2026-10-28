"""Check routing, captured input refresh, checkpoint keys and native fallback."""
import torch
from unifolm_wma.utils.triton_compensated_linear import (
    compensated_linear, make_forward, install_compensated_linears)


@torch.no_grad()
def main():
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    for inputs,bias in ((320,False),(320,True),(1280,True)):
        layer=torch.nn.Linear(inputs,320,bias=bias).cuda().eval()
        value=torch.randn(2560,16,inputs,device='cuda')
        original=layer.forward; candidate=make_forward(layer,narrow_output=True)
        keys=tuple(layer.state_dict())
        torch.testing.assert_close(candidate(value),
            compensated_linear(value,layer.weight,layer.bias),rtol=0,atol=0)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):candidate(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=candidate(value)
        value.add_(.1);graph.replay()
        torch.testing.assert_close(captured,candidate(value),rtol=0,atol=0)
        for x in (value[:1],value.transpose(0,1).contiguous().transpose(0,1)):
            torch.testing.assert_close(candidate(x),original(x),rtol=0,atol=0)
        layer.train()
        torch.testing.assert_close(candidate(value),original(value),rtol=0,atol=0)
        layer.eval()
        with torch.enable_grad():
            x=value[:1].detach().requires_grad_(True)
            a,b=candidate(x),original(x)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(torch.autograd.grad(a.sum(),x)[0],
                torch.autograd.grad(b.sum(),x)[0],rtol=0,atol=0)
        assert keys==tuple(layer.state_dict())
        layer.half(); x=value.half()
        torch.testing.assert_close(candidate(x),original(x),rtol=0,atol=0)
        print('NARROW_OUTPUT_GUARDS inputs='+str(inputs)+' bias='+str(bias)+' passed=true',flush=True)
    model=torch.nn.Sequential(torch.nn.Linear(320,320),torch.nn.Linear(640,640)).eval()
    keys=tuple(model.state_dict())
    counts=install_compensated_linears(model,narrow_output=True)
    assert counts=={'Linear':2,'NarrowOutputLinear':1}
    assert keys==tuple(model.state_dict())
    x=torch.randn(2,320)
    torch.testing.assert_close(model[0](x),torch.nn.functional.linear(
        x,model[0].weight,model[0].bias),rtol=0,atol=0)
    print('NARROW_OUTPUT_LINEAR_CHECK passed=true full_video_quality=unverified',flush=True)


if __name__=='__main__':main()
