"""Real blocks and captured stages sharing GEGLU and old decision final caches."""
import gc
import json
import os
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from unifolm_wma.modules.attention import BasicTransformerBlock,GEGLU
from unifolm_wma.utils import decision_primary_hidden_parts as helper
from unifolm_wma.utils import world_norm3_packed_geglu as world_helper
from unifolm_wma.utils import decision_ffn_residual_epilogue as old_helper
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_compensated_linear import make_forward as compensated
from unifolm_wma.utils.triton_prepacked_geglu import prepacked_geglu
from unifolm_wma.utils.triton_primary_geglu import primary_geglu
from unifolm_wma.utils.triton_prepacked_input_linear import prepacked_input_linear
from unifolm_wma.utils.decision_primary_ffn import primary_linear
from unifolm_wma.utils.triton_residual_layer_norm import fused_residual_layer_norm


def exact(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype
 assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


class AffineAttention(torch.nn.Module):
 def forward(self,value,context=None,mask=None):
  output=value*.125+.01
  if context is not None:output=output+context.mean()*.00001
  return output


def configure(block):
 block._inference_fused_residual_norm=True;block._world_norm3_packed_geglu_installed=True
 block._decision_ffn_residual_epilogue_installed=True
 block.ff.net[0]._prepacked_geglu_installed=True;block.ff.net[0]._decision_primary_geglu_installed=True
 for key in ['_tf32x3_installed','_decision_vendor_ffn_installed','_decision_primary_ffn_installed']:setattr(block.ff.net[2],key,True)


def fixture(bias):
 block=BasicTransformerBlock(1280,20,64,checkpoint=False).cuda().eval()
 block.attn1=AffineAttention().eval();block.attn2=AffineAttention().eval()
 if not bias:
  block.ff.net[0].proj.register_parameter('bias',None);block.ff.net[2].register_parameter('bias',None)
 configure(block);ge=block.ff.net[0];proj=ge.proj;final=block.ff.net[2]
 world_final=compensated(final,min_matmul_macs=0,backend='scaled-fp16',cache_half_weights=True,group_m=32,block_k=64,pack_inputs=True)
 def original_geglu(value):
  parts=pack_half_weight(proj.weight)
  if torch.backends.cudnn.allow_tf32:return primary_geglu(value,parts,proj.bias,mode='both')
  return prepacked_geglu(value,parts,proj.bias,block_m=64,block_n=32,group_m=32,num_warps=4,num_stages=2)
 def original_final(value):
  if torch.backends.cudnn.allow_tf32:return primary_linear(value,pack_half_weight(final.weight)[0],final.bias)
  return world_final(value)
 ge.forward=original_geglu;final.forward=original_final
 block._forward=world_helper.make_forward(block)
 block._forward=old_helper.make_forward(block)
 return block


def golden(block,value,context):
 attended=block.attn1(block.norm1(value),context=context if block.disable_self_attn else None)
 residual,normalized=fused_residual_layer_norm(attended,value,block.norm2)
 attended=block.attn2(normalized,context=context)
 residual,normalized=fused_residual_layer_norm(attended,residual,block.norm3)
 ge=block.ff.net[0].proj;final=block.ff.net[2];gp=pack_half_weight(ge.weight);fp=pack_half_weight(final.weight)
 if torch.backends.cudnn.allow_tf32:
  hidden=primary_geglu(normalized,gp,ge.bias,mode='both')
  result=primary_linear(hidden,fp[0],final.bias)
 else:
  hidden=prepacked_geglu(normalized,gp,ge.bias,block_m=64,block_n=32,group_m=32,num_warps=4,num_stages=2)
  result=prepacked_input_linear(hidden,final.weight,final.bias,weight_parts=fp,block_m=64 if value.numel()//1280>=2560 else 32,block_n=64,block_k=64,group_m=32,num_warps=4,num_stages=2)
 assert result is not None
 return result+residual


def sentinel(value,context=None,mask=None):return value*.75+.25


@torch.no_grad()
def main():
 torch.manual_seed(1133);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
 os.environ.update(UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU='all',UNIFOLM_NORM_BUFFER_NOFILL='all',UNIFOLM_GEMM_OUTPUT_NOFILL='all',UNIFOLM_HALF_PACK_NOFILL='1',UNIFOLM_GEGLU_OUTPUT_NOFILL='1')
 for rows in [2560,640]:
  for bias in [False,True]:
   for first in ['world','decision']:
    block=fixture(bias);ge=block.ff.net[0];proj=ge.proj;final=block.ff.net[2]
    proj.weight.copy_(proj.weight.half().float());final.weight.copy_(final.weight.half().float())
    keys=set(block.state_dict());masters=[p.data_ptr() for p in block.parameters()]
    old_forward=block._forward;forward=helper.make_forward(block)
    value=torch.randn(16,rows//16,1280,device='cuda')*.3;context=torch.randn(1,1,1280,device='cuda')*.1
    packs=[];pack=helper.pack_half_weight
    def recorder(stage):
     def recorded(*args,**kwargs):
      result=pack(*args,**kwargs);packs.append(stage);return result
     return recorded
    def call(stage,fn=forward):
     torch.backends.cudnn.allow_tf32=stage=='decision'
     return fn(value,context=context)
    def pointers():
     cache=ge._norm3_geglu_weight_cache
     return tuple(p.data_ptr() for p in cache[1]),cache[2].data_ptr(),tuple(p.data_ptr() for p in final._decision_ffn_residual_weight_cache[1])
    with patch.object(helper,'pack_half_weight',side_effect=recorder('new_decision')),patch.object(world_helper,'pack_half_weight',side_effect=recorder('world')),patch.object(old_helper,'pack_half_weight',side_effect=recorder('old_decision')):
     assert not hasattr(ge,'_norm3_geglu_weight_cache') and not hasattr(final,'_decision_ffn_residual_weight_cache')
     for stage in [first,'decision' if first=='world' else 'world']:exact(call(stage),golden(block,value,context))
     assert len(packs)==2 and packs==(['world','new_decision'] if first=='world' else ['new_decision','new_decision'])
     storage=pointers();assert int(ge._norm3_geglu_weight_cache[2])==0
     for stage in ['world','decision']:
      for _ in range(2):exact(call(stage),golden(block,value,context))
     assert len(packs)==2
     stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(stream):
      for stage in ['world','decision']:
       for _ in range(3):call(stage)
     torch.cuda.current_stream().wait_stream(stream);graphs={};captured={}
     for stage in ['world','decision']:
      torch.backends.cudnn.allow_tf32=stage=='decision';graphs[stage]=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graphs[stage],stream=stream):captured[stage]=forward(value,context=context)
     assert len(packs)==2
     for quantized in [False,True]:
      value.add_(.0001);context.add_(.0002);proj.weight.add_(.00013);final.weight.add_(.00013)
      if quantized:proj.weight.copy_(proj.weight.half().float());final.weight.copy_(final.weight.half().float())
      if proj.bias is not None:proj.bias.add_(.0002);final.bias.add_(.0001)
      for name in ['norm1','norm2','norm3']:
       norm=getattr(block,name);norm.weight.mul_(1.001);norm.bias.add_(.0003)
      # Each stage refreshes its own final cache; only GE parts cross stages.
      for stage in ['world','decision']:exact(call(stage),golden(block,value,context))
      assert pointers()==storage and (int(ge._norm3_geglu_weight_cache[2])==0)==quantized
      for stage in ['world','decision']:
       torch.backends.cudnn.allow_tf32=stage=='decision';graphs[stage].replay();exact(captured[stage],golden(block,value,context))
     assert len(packs)==6
     proj.weight.add_(.000043);call('decision');assert len(packs)==7 and packs[-1]=='new_decision'
     torch.backends.cudnn.allow_tf32=False;graphs['world'].replay();exact(captured['world'],golden(block,value,context))
     assert int(ge._norm3_geglu_weight_cache[2])!=0
     proj.weight.add_(.000057);proj.weight.copy_(proj.weight.half().float());call('world')
     assert len(packs)==8 and packs[-1]=='world' and int(ge._norm3_geglu_weight_cache[2])==0
     torch.backends.cudnn.allow_tf32=True;graphs['decision'].replay();exact(captured['decision'],golden(block,value,context))
     # Warm and capture the old decision GE path only after GE mutations;
     # the following tests change final weights alone.
     call('decision',old_forward);stream.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(stream):
      for _ in range(3):call('decision',old_forward)
     torch.cuda.current_stream().wait_stream(stream);oldgraph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(oldgraph,stream=stream):oldout=old_forward(value,context=context)
     final.weight.add_(.000043)
     if final.bias is not None:final.bias.add_(.00017)
     call('decision',old_forward);assert len(packs)==9 and packs[-1]=='old_decision'
     graphs['decision'].replay();exact(captured['decision'],golden(block,value,context))
     final.weight.add_(.000057)
     if final.bias is not None:final.bias.add_(.00019)
     call('decision');assert len(packs)==10 and packs[-1]=='new_decision'
     oldgraph.replay();exact(oldout,golden(block,value,context));assert pointers()==storage
    assert keys==set(block.state_dict()) and masters==[p.data_ptr() for p in block.parameters()] and all(p.dtype==torch.float32 for p in block.parameters())
    torch.backends.cudnn.allow_tf32=True;block._forward=sentinel;fallback=helper.make_forward(block)
    with patch.object(helper,'residual_norm_half_pack',side_effect=AssertionError('guard entered fusion')):
     torch.backends.cudnn.allow_tf32=False;exact(fallback(value),sentinel(value));torch.backends.cudnn.allow_tf32=True
     torch.backends.cuda.matmul.allow_tf32=True;exact(fallback(value),sentinel(value));torch.backends.cuda.matmul.allow_tf32=False
     block.train();exact(fallback(value),sentinel(value));block.eval()
     with torch.enable_grad():exact(fallback(value),sentinel(value))
     with torch.autocast('cuda',dtype=torch.float16):exact(fallback(value),sentinel(value))
     for other in [value[:1],value.half(),value.transpose(1,2).contiguous().transpose(1,2),value.cpu()]:
      left=fallback(other);right=sentinel(other);assert left.dtype==right.dtype and torch.equal(left,right)
     block.ff._inference_dtype=torch.float16;exact(fallback(value),sentinel(value));block.ff._inference_dtype=torch.float32
     block.norm3.train();exact(fallback(value),sentinel(value));block.norm3.eval()
    for kind in ['both','only_GE_warm','only_final_warm','GE_stale','final_stale']:
     cold=fixture(bias);cold._forward=sentinel;cold_forward=helper.make_forward(cold)
     if kind=='only_GE_warm':
      torch.backends.cudnn.allow_tf32=False;world_helper.make_forward(cold)(value);torch.backends.cudnn.allow_tf32=True
     if kind=='only_final_warm':old_helper.make_forward(cold)(value)
     if kind.endswith('_stale'):
      cold_forward(value);p=cold.ff.net[0].proj if kind=='GE_stale' else cold.ff.net[2];p.weight.add_(.00013)
     with patch.object(helper,'pack_half_weight',side_effect=AssertionError('capture packed weights')):
      cg=torch.cuda.CUDAGraph()
      with torch.cuda.graph(cg):co=cold_forward(value)
      cg.replay();exact(co,sentinel(value))
     if kind in ['both','only_final_warm']:assert not hasattr(cold.ff.net[0],'_norm3_geglu_weight_cache')
     if kind in ['both','only_GE_warm']:assert not hasattr(cold.ff.net[2],'_decision_ffn_residual_weight_cache')
     del cold,cold_forward,cg,co
    with patch.object(helper,'_READY',set()):
     readygraph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(readygraph):readyout=fallback(value)
     readygraph.replay();exact(readyout,sentinel(value))
    print('DECISION_PRIMARY_HIDDEN_PARTS_RUNTIME '+json.dumps(dict(rows=rows,bias=bias,first_stage=first,
     real_block_independent_both_stage_reference_signedzero=True,first_timed_shared_caches_both_warm_orders=True,
     graph_inputs_context_all_norms_GE_final_masters_bias=True,GE_cache_refresh_decision_to_world_graph=True,
     GE_cache_refresh_world_to_decision_graph=True,old_new_decision_final_cache_cross_refresh=True,
     world_final_distinct_cache_refreshed_on_own_stage=True,version_refresh_same_storage=True,zero_nonzero_zero_GE_predicate=True,
     fp32_master_state_preserved=True,cold_partial_and_each_stale_capture_fallback=True,missing_cublas_ready_fallback=True,native_fallbacks=11)),flush=True)
    del block,ge,proj,final,old_forward,forward,value,context,graphs,captured,oldgraph,oldout,fallback,readygraph,readyout
    gc.collect();torch.cuda.empty_cache()
 with torch.device('meta'):
  tree=torch.nn.Module();tree.wide=torch.nn.ModuleList([BasicTransformerBlock(1280,20,64) for _ in range(12)])
  tree.low=BasicTransformerBlock(320,5,64);tree.mid=BasicTransformerBlock(640,10,64);tree.initial=BasicTransformerBlock(512,8,64);tree.outside=GEGLU(1280,5120)
 for block in tree.wide:configure(block)
 assert helper.install_decision_primary_hidden_parts(tree)==12
 assert all(not getattr(m,'_decision_primary_hidden_parts_installed',False) for m in [tree.low,tree.mid,tree.initial,tree.outside])
 try:helper.install_decision_primary_hidden_parts(tree)
 except ValueError:pass
 else:raise AssertionError('duplicate installation accepted')
 print('DECISION_PRIMARY_HIDDEN_PARTS_RUNTIME_CHECK passed=true fixtures=8 both_stage_GE_and_old_new_decision_final_cross_refresh=true fallback11_coldpartialstale5_ready=true factory12=true duplicate_rejected=true')


if __name__=='__main__':main()
