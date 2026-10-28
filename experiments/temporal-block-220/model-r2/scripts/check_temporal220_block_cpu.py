"""Exercise grouped row ownership and both live predicate orders on CPU."""
from collections import Counter
import json
from types import SimpleNamespace
import torch
from temporal220_block_mode import grouped_route,dispatch
assert not torch.cuda.is_initialized()
original_capture=torch.cuda.is_current_stream_capturing
torch.cuda.is_current_stream_capturing=lambda:False
cases=0
try:
 for m,c in ((640,1280),(2560,1280),(10240,640),(40960,320)):
  x=torch.empty((m,c));y=torch.full((3,m,c),-19.);scratch=torch.empty((3,m//2,c))
  for rank in (0,1):
   values=dict(XH=x,XL=x,Y=y,M=m,N=3*c,K=c,BM=64,BN=64,GROUP=32,OUTPUT_GROUPS=3,THREE=False)
   routed,grid=grouped_route(values,rank,scratch)
   assert routed['XH'].storage_offset()==rank*(m//2)*c and routed['M']==m//2 and routed['Y'] is scratch
   assert {k:v for k,v in routed.items() if not isinstance(v,torch.Tensor) and k!='M'}=={k:v for k,v in values.items() if not isinstance(v,torch.Tensor) and k!='M'}
   assert grid==((m//2+63)//64*((3*c+63)//64),)
   def kernel():pass
   kernel.__name__='_zero_low_gemm_persistent';jit=SimpleNamespace(arg_names=list(values),fn=kernel)
   for live_three in (False,True):
    y.fill_(-19.);scope=dict(M=m,C=c,scratch={},shaders=Counter(),grouped_copy_nodes=0)
    def parent(jit,*args,**kwargs):
     v=dict(zip(jit.arg_names,args));assert v['M']==m//2
     if v['THREE']==live_three:v['Y'].fill_(37.)
    for three in (False,True):
     values['THREE']=three
     dispatch(jit,tuple(values[n] for n in jit.arg_names),dict(grid=(216,),enable_fp_fusion=False),scope,rank,parent)
    owned=y[:,rank*(m//2):(rank+1)*(m//2)];other=y[:,(1-rank)*(m//2):(2-rank)*(m//2)]
    assert torch.all(owned==37.) and torch.all(other==-19.) and len(scope['scratch'])==1
    cases+=1
 del x,y,scratch
finally:torch.cuda.is_current_stream_capturing=original_capture
assert not torch.cuda.is_initialized()
print('TEMPORAL220_BLOCK_CPU '+json.dumps(dict(passed=True,cases=cases,no_GPU_initialized=True,both_live_predicate_orders=True,unowned_rows_unchanged=True)))
