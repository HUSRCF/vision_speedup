"""Real CrossAttention output/cache/graph contracts with an independent reference."""
import json
import os
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.attention import CrossAttention,SpatialTransformer
from unifolm_wma.utils import world_attention_hidden_projection as helper
from unifolm_wma.utils import world_qkv_packed_attention as previous
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_zero_weight_low_linear import weight_low_count
from unifolm_wma.utils.triton_qkv_packed_attention import qkv_packed_attention
from unifolm_wma.utils.triton_two_product_linear import two_product_linear


def exact(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype
 bits=torch.int32 if a.dtype==torch.float32 else torch.int16
 assert torch.equal(a.contiguous().view(bits),b.contiguous().view(bits))


def configure(module):
 module._compensated_spatial_attention=True;module._world_zero_weight_low_qkv_installed=True
 module._compensated_self_qkv_enabled=True;module._inference_attention_layout='bmhk'
 module._inference_attention_dtype=module._inference_projection_dtype=torch.float32
 module._qkv_packed_attention_installed=True
 setattr(module.to_out[0], '_world_two_product_output_installed' if module.to_q.in_features==320 else '_world_two_product_roles_installed',True)


def sentinel(value,context=None,mask=None):return value*.75+.25


def fixture(channels,bias):
 module=CrossAttention(channels,heads=channels//64,dim_head=64).cuda().eval()
 if not bias:module.to_out[0].register_parameter('bias',None)
 configure(module);module.forward=sentinel
 return module


def reference(module,value):
 projections=module.to_q,module.to_k,module.to_v
 combined=torch.cat([p.weight for p in projections],dim=0)
 qp=pack_half_weight(combined);count=weight_low_count(qp)
 output=qkv_packed_attention(value,qp,count,scale=.125)
 final=module.to_out[0]
 return two_product_linear(output,pack_half_weight(final.weight),final.bias,protect='input')


def pointers(module):
 cache=module._world_zero_weight_low_qkv_cache;fc=module.to_out[0]._attention_hidden_projection_weight_cache
 return cache[1].data_ptr(),tuple(p.data_ptr() for p in cache[2]),cache[3].data_ptr(),tuple(p.data_ptr() for p in fc[1])


def final_cache_key(weight):return weight.data_ptr(),weight._version,tuple(weight.shape),weight.dtype,weight.device


def seed_final(module):
 weight=module.to_out[0].weight
 module.to_out[0]._attention_hidden_projection_weight_cache=final_cache_key(weight),pack_half_weight(weight)


@torch.no_grad()
def real_guards():
 torch.manual_seed(1129);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 os.environ.update(UNIFOLM_GEMM_OUTPUT_NOFILL='all',UNIFOLM_HALF_PACK_NOFILL='1',UNIFOLM_ATTENTION_OUTPUT_NOFILL='all')
 for channels,length in [(320,2560),(640,640)]:
  for bias in [False,True]:
   module=fixture(channels,bias);projections=module.to_q,module.to_k,module.to_v;final=module.to_out[0]
   for p in projections:p.weight.copy_(p.weight.half().float())
   keys=set(module.state_dict());masters=[p.data_ptr() for p in module.parameters()]
   forward=helper.make_forward(module);value=torch.randn(16,length,channels,device='cuda')*.3
   packs=[];pack=helper.pack_half_weight
   def recorded(*args,**kwargs):
    result=pack(*args,**kwargs);packs.append(tuple(p.data_ptr() for p in result));return result
   with patch.object(helper,'pack_half_weight',side_effect=recorded):
    assert not hasattr(module,'_world_zero_weight_low_qkv_cache') and not hasattr(final,'_attention_hidden_projection_weight_cache')
    actual=forward(value);exact(actual,reference(module,value));assert len(packs)==2
    storage=pointers(module);assert int(module._world_zero_weight_low_qkv_cache[3])==0
    for _ in range(3):exact(actual,forward(value))
    assert len(packs)==2
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     for _ in range(3):forward(value)
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=forward(value)
    assert len(packs)==2
    for quantized in [False,True]:
     value.add_(.0001)
     for p in projections:
      p.weight.add_(.00013)
      if quantized:p.weight.copy_(p.weight.half().float())
     final.weight.add_(.00013)
     if final.bias is not None:final.bias.add_(.0001)
     eager=forward(value);exact(eager,reference(module,value))
     assert pointers(module)==storage and (int(module._world_zero_weight_low_qkv_cache[3])==0)==quantized
     graph.replay();exact(captured,reference(module,value))
    assert len(packs)==6
    for layer in [*projections,final]:
     before=len(packs);layer.weight.add_(.00013);forward(value);assert len(packs)==before+1
     graph.replay();exact(captured,reference(module,value));assert pointers(module)==storage
    assert len(packs)==10
    for p in projections:p.weight.copy_(p.weight.half().float())
    forward(value);assert len(packs)==11 and int(module._world_zero_weight_low_qkv_cache[3])==0
    graph.replay();exact(captured,reference(module,value))
    # The accepted wrapper refreshes the same QKV buffers; the new captured
    # reader must observe them without running its Python refresh path.
    before=len(packs);module.to_q.weight.add_(.00013)
    previous.make_forward(module)(value)
    assert len(packs)==before and pointers(module)==storage
    graph.replay();exact(captured,reference(module,value))
   assert keys==set(module.state_dict()) and masters==[p.data_ptr() for p in module.parameters()]
   assert all(p.dtype==torch.float32 for p in module.parameters())
   with patch.object(helper,'qkv_attention_hidden_projection',side_effect=AssertionError('guard entered fusion')):
    torch.backends.cudnn.allow_tf32=True;exact(forward(value),sentinel(value));torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=True;exact(forward(value),sentinel(value));torch.backends.cuda.matmul.allow_tf32=False
    module.train();exact(forward(value),sentinel(value));module.eval()
    with torch.enable_grad():exact(forward(value),sentinel(value))
    with torch.autocast('cuda',dtype=torch.float16):exact(forward(value),sentinel(value))
    for other in [value[:1],value.half(),value.transpose(1,2).contiguous().transpose(1,2),value.cpu()]:exact(forward(other),sentinel(other))
    exact(forward(value,context=value),sentinel(value));exact(forward(value,mask=torch.ones(1,device='cuda')),sentinel(value))
    module.to_out[1].train();exact(forward(value),sentinel(value));module.eval()
    weight=final.weight;final.weight=torch.nn.Parameter(weight.t());exact(forward(value),sentinel(value));final.weight=weight
   for kind in ['both_cold','only_qkv_warm','only_output_warm','qkv_stale','output_stale']:
    cold=fixture(channels,bias);cold_forward=helper.make_forward(cold)
    if kind=='only_qkv_warm':previous.make_forward(cold)(value)
    if kind=='only_output_warm':seed_final(cold)
    if kind.endswith('_stale'):
     cold_forward(value);layer=cold.to_q if kind=='qkv_stale' else cold.to_out[0];layer.weight.add_(.00013)
    with patch.object(helper,'pack_half_weight',side_effect=AssertionError('capture packed weights')):
     cg=torch.cuda.CUDAGraph()
     with torch.cuda.graph(cg):co=cold_forward(value)
     cg.replay();exact(co,sentinel(value))
    if kind=='both_cold':assert not hasattr(cold,'_world_zero_weight_low_qkv_cache') and not hasattr(cold.to_out[0],'_attention_hidden_projection_weight_cache')
    if kind=='only_qkv_warm':assert not hasattr(cold.to_out[0],'_attention_hidden_projection_weight_cache')
    if kind=='only_output_warm':assert not hasattr(cold,'_world_zero_weight_low_qkv_cache')
    del cold,cold_forward,cg,co
   # A hot accepted QKV cache is reused; a pre-existing output cache is also
   # reused when QKV has not been prepared yet. Both cases pack one weight.
   for warm in ['qkv','output']:
    partial=fixture(channels,bias)
    if warm=='qkv':previous.make_forward(partial)(value)
    else:seed_final(partial)
    before=[]
    with patch.object(helper,'pack_half_weight',side_effect=lambda *a,**kw:(before.append(True) or pack(*a,**kw))):
     exact(helper.make_forward(partial)(value),reference(partial,value));assert len(before)==1
    del partial
   print('WORLD_ATTENTION_HIDDEN_PROJECTION_RUNTIME '+json.dumps(dict(channels=channels,bias=bias,real_CrossAttention_reference_signedzero=True,
    first_timed_two_caches_and_partial_warm_reuse=True,each_QKV_and_output_master_version_refresh_same_storage=True,
    zero_nonzero_zero_predicate_graph=True,input_three_QKV_output_master_bias_graph=True,old_wrapper_refresh_new_graph_shared_QKV=True,
    fp32_master_state_preserved=True,cold_partial_and_each_stale_capture_fallback=True,native_fallbacks=13)),flush=True)
   del module,projections,final,forward,value,actual,eager,graph,captured
   torch.cuda.empty_cache()
 for scope,count in [('all',10),('c320',5),('c640',5)]:
  with torch.device('meta'):
   tree=torch.nn.Module();tree.low=torch.nn.ModuleList([SpatialTransformer(320,5,64,depth=1) for _ in range(5)])
   tree.mid=torch.nn.ModuleList([SpatialTransformer(640,10,64,depth=1) for _ in range(5)])
   tree.initial=SpatialTransformer(512,8,64,depth=1);tree.wide=SpatialTransformer(1280,20,64,depth=1)
   tree.outside=CrossAttention(320,heads=5,dim_head=64)
  for parent in [*tree.low,*tree.mid]:
   for block in parent.transformer_blocks:configure(block.attn1);configure(block.attn2)
  configure(tree.outside)
  assert helper.install_world_attention_hidden_projection(tree,scope)==count
  assert not getattr(tree.outside,'_world_attention_hidden_projection_installed',False)
  assert all(not getattr(b.attn2,'_world_attention_hidden_projection_installed',False) for p in [*tree.low,*tree.mid] for b in p.transformer_blocks)
  assert all(not getattr(b.attn1,'_world_attention_hidden_projection_installed',False) for p in [tree.initial,tree.wide] for b in p.transformer_blocks)
  try:helper.install_world_attention_hidden_projection(tree,scope)
  except ValueError:pass
  else:raise AssertionError('duplicate installation accepted')
 print('WORLD_ATTENTION_HIDDEN_PROJECTION_RUNTIME_CHECK passed=true fixtures=4 real_factory_scope_counts=10,5,5 duplicate_rejected=true')


@torch.no_grad()
def main():
 from contextlib import nullcontext
 from attention_projection_schedule_runtime_hooks import installed
 from unifolm_wma.utils.attention_hidden_parts_pipeline import packed_attention_output
 from unifolm_wma.utils import gemm_output_allocation as buffers
 mode='UNIFOLM_WORLD_ATTENTION_PROJECTION_SCHEDULE';os.environ[mode]='1'
 with installed() as launches:
  real_guards()
  inputs={}
  for name,shape,channels in [('c320',(16,2560,5,64),320),('c640',(16,640,10,64),640),
      ('batch',(1,2560,5,64),320),('length',(16,13,5,64),320),('channels',(2,73,3,64),71),('factorization',(32,1280,5,64),320)]:
   value=torch.randn(*shape,device='cuda')*.3
   high,low=pack_half_weight(value.reshape(-1,shape[2]*64),weight_scale=1.)
   hidden=high.view(shape),low.view(shape)
   pp=pack_half_weight(torch.randn(channels,shape[2]*64,device='cuda')*.02)
   bias=torch.randn(channels,device='cuda')*.02
   inputs[name]=(hidden,pp,bias)
  for policy in ['selected_c320','selected_c640','disabled','decision','matmul_tf32','no_bias','other_batch','other_length','other_channels','other_factorization','grad_enabled']:
   torch.backends.cudnn.allow_tf32=policy=='decision';torch.backends.cuda.matmul.allow_tf32=policy=='matmul_tf32'
   name='c640' if policy=='selected_c640' else policy.removeprefix('other_') if policy.startswith('other_') else 'c320'
   hidden,parts,bias=inputs[name]
   if policy=='no_bias':bias=None
   with torch.enable_grad() if policy=='grad_enabled' else nullcontext():
    os.environ[mode]='0';expected=packed_attention_output(hidden,parts,bias)
    os.environ[mode]='0' if policy=='disabled' else '1'
    before=len(launches);old=buffers._debug_poison
    try:
     buffers._debug_poison=float('nan');actual=packed_attention_output(hidden,parts,bias)
    finally:buffers._debug_poison=old
    exact(actual,expected);assert torch.isfinite(actual).all()
    rr=launches[before:];assert len(rr)==1 and rr[0]['selected']==policy.startswith('selected_')
    fresh=packed_attention_output(hidden,parts,bias);assert fresh.data_ptr()!=actual.data_ptr();exact(fresh,expected)
  os.environ[mode]='invalid'
  try:packed_attention_output(*inputs['c320'])
  except ValueError as e:assert 'attention projection schedule' in str(e)
  else:raise AssertionError('Invalid attention projection schedule accepted')
  os.environ[mode]='1';torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 print('ATTENTION_PROJECTION_SCHEDULE_RUNTIME '+json.dumps(dict(passed=True,real_block_fixtures=4,policy_fixtures=12,
  original_real_wrapper_cache_factory_and_13_fallback_guards_retained=True,first_timed_master_cache_same_storage=True,
  input_three_QKV_output_master_bias_graph=True,native_BK64_FP32_signedzero=True,NaN=True,fresh_output=True,
  mode_backend_shape_bias_grad_fallbacks=True,invalid_mode_rejected=True,actual_kernel_launches=len(launches))),flush=True)
 print('ATTENTION_PROJECTION_SCHEDULE_RUNTIME_CHECK passed=true fixtures=4 policies=12',flush=True)


if __name__=='__main__':main()

