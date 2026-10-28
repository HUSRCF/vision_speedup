"""Real FeedForward stages, cache lifecycle, graph mutation and dispatch guards."""
import json
import os
from unittest.mock import patch
import torch
from unifolm_wma.modules.attention import FeedForward
from unifolm_wma.utils import world_initial_native_hidden_parts as helper
from unifolm_wma.utils import initial_bk32_native_geglu as native
from unifolm_wma.utils import half_hidden_initial_ffn as decision
from unifolm_wma.utils.triton_compensated_linear import make_forward as compensated


def exact(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype
 bits=torch.int16 if a.dtype==torch.float16 else torch.int32
 assert torch.equal(a.contiguous().view(bits),b.contiguous().view(bits))


def configure(module):
 module._half_hidden_initial_ffn_installed=True
 module.net[0]._initial_bk32_native_geglu_installed=True
 for p in [module.net[0].proj,module.net[2]]:
  p._initial_world_linear_installed=True;p._decision_initial_primary_installed=True


def fixture(bias):
 module=FeedForward(512,glu=True).cuda().eval()
 if not bias:
  module.net[0].proj.register_parameter('bias',None);module.net[2].register_parameter('bias',None)
 configure(module)
 for p in [module.net[0].proj,module.net[2]]:
  p.forward=compensated(p,min_matmul_macs=0,backend='scaled-fp16',cache_half_weights=True,group_m=32,block_k=32,pack_inputs=False)
 module.net[0].forward=native.make_forward(module.net[0])
 module.forward=decision.make_forward(module)
 return module


def sentinel(value):return value*.75+.25


@torch.no_grad()
def main():
 torch.manual_seed(1125);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 os.environ.update(UNIFOLM_GEMM_OUTPUT_NOFILL='all',UNIFOLM_HALF_PACK_NOFILL='1',UNIFOLM_GEGLU_OUTPUT_NOFILL='1')
 for bias in [False,True]:
  for first_stage in ['world','decision']:
   module=fixture(bias);original=module.forward;forward=helper.make_forward(module)
   keys=set(module.state_dict());masters=[p.data_ptr() for p in module.parameters()]
   x=torch.randn(2560,16,512,device='cuda')*.3
   packs=[];pack=helper.pack_half_weight
   with patch.object(helper,'pack_half_weight',side_effect=lambda *a,**kw:(packs.append(kw.get('out')) or pack(*a,**kw))):
    assert not hasattr(module,'_world_initial_native_hidden_parts_weight_cache') and not packs
    torch.backends.cudnn.allow_tf32=first_stage=='decision';exact(forward(x),original(x))
    assert len(packs)==(0 if first_stage=='decision' else 2)
    torch.backends.cudnn.allow_tf32=False;exact(forward(x),original(x));assert len(packs)==2
    cache=module._world_initial_native_hidden_parts_weight_cache;pointers=tuple(p.data_ptr() for parts in cache[1] for p in parts)
    for _ in range(2):exact(forward(x),original(x))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     for _ in range(3):forward(x)
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=forward(x)
    assert len(packs)==2
    for quantized in [False,True]:
     x.add_(.001)
     for p in [module.net[0].proj,module.net[2]]:
      p.weight.add_(.00013)
      if quantized:p.weight.copy_(p.weight.half().float())
      if p.bias is not None:p.bias.add_(.0002)
     eager=forward(x);exact(eager,original(x));graph.replay();exact(captured,original(x))
     assert pointers==tuple(p.data_ptr() for parts in module._world_initial_native_hidden_parts_weight_cache[1] for p in parts)
    assert len(packs)==6
    for p in [module.net[0].proj,module.net[2]]:
     before=len(packs);p.weight.add_(.0001);forward(x);assert len(packs)==before+1
     graph.replay();exact(captured,original(x))
    assert len(packs)==8
    torch.backends.cudnn.allow_tf32=True
    before=len(packs);exact(forward(x),original(x));assert len(packs)==before
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     for _ in range(3):forward(x)
    torch.cuda.current_stream().wait_stream(stream);dg=torch.cuda.CUDAGraph()
    with torch.cuda.graph(dg,stream=stream):decision_out=forward(x)
    x.add_(.001);original(x);dg.replay();exact(decision_out,original(x))
    torch.backends.cudnn.allow_tf32=False
   assert keys==set(module.state_dict()) and masters==[p.data_ptr() for p in module.parameters()]
   assert all(p.dtype==torch.float32 for p in module.parameters())
   module.forward=sentinel;fallback=helper.make_forward(module)
   with patch.object(helper,'initial_native_hidden_parts',side_effect=AssertionError('guard entered producer')):
    torch.backends.cudnn.allow_tf32=True;exact(fallback(x),sentinel(x));torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=True;exact(fallback(x),sentinel(x));torch.backends.cuda.matmul.allow_tf32=False
    module.train();exact(fallback(x),sentinel(x));module.eval()
    module.net[0].proj.train();exact(fallback(x),sentinel(x));module.eval()
    with torch.enable_grad():exact(fallback(x),sentinel(x))
    with torch.autocast('cuda',dtype=torch.float16):exact(fallback(x),sentinel(x))
    for other in [x[:1],x.half(),x.transpose(0,1).contiguous().transpose(0,1),x.cpu()]:exact(fallback(other),sentinel(other))
    module._inference_dtype=torch.float16;exact(fallback(x),sentinel(x));module._inference_dtype=torch.float32
    module._half_hidden_initial_ffn_installed=False;exact(fallback(x),sentinel(x));module._half_hidden_initial_ffn_installed=True
    module.net[1].train();exact(fallback(x),sentinel(x));module.eval()
   for kind in ['cold','geglu_stale','final_stale']:
    cold=fixture(bias);cold.forward=sentinel;cold_forward=helper.make_forward(cold)
    if kind!='cold':
     cold_forward(x);target=cold.net[0].proj if kind=='geglu_stale' else cold.net[2];target.weight.add_(.00013)
    with patch.object(helper,'pack_half_weight',side_effect=AssertionError('capture packed weights')):
     cg=torch.cuda.CUDAGraph()
     with torch.cuda.graph(cg):co=cold_forward(x)
     cg.replay();exact(co,sentinel(x))
    if kind=='cold':assert not hasattr(cold,'_world_initial_native_hidden_parts_weight_cache')
    del cold,cold_forward,cg,co
   print('WORLD_INITIAL_NATIVE_HIDDEN_RUNTIME '+json.dumps(dict(bias=bias,first_stage=first_stage,real_feedforward_both_stages_signedzero=True,
    two_caches_first_timed_warm=True,version_refresh_same_storage=True,unchanged_weight_not_repacked=True,
    graph_input_two_masters_bias=True,decision_fallthrough_graph=True,fp32_master_state_preserved=True,
    cold_and_each_stale_capture_fallback=True,native_fallbacks=13)),flush=True)
   del module,original,forward,x,graph,captured,eager,dg,decision_out,fallback,cache
   torch.cuda.empty_cache()
 with torch.device('meta'):
  tree=torch.nn.Module();tree.init_attn=torch.nn.ModuleList([FeedForward(512,glu=True)])
  tree.other=FeedForward(512,glu=True);tree.init_attn.append(FeedForward(320,glu=True))
 configure(tree.init_attn[0]);assert helper.install_world_initial_native_hidden_parts(tree)==1
 assert not getattr(tree.other,'_world_initial_native_hidden_parts_installed',False)
 assert not getattr(tree.init_attn[1],'_world_initial_native_hidden_parts_installed',False)
 try:helper.install_world_initial_native_hidden_parts(tree)
 except ValueError:pass
 else:raise AssertionError('duplicate accepted')
 print('WORLD_INITIAL_NATIVE_HIDDEN_RUNTIME_CHECK passed=true fixtures=4 factory_scope=true duplicate_rejected=true')


if __name__=='__main__':main()
