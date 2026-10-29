"""Reject unsafe binding before any live device test; CUDA stays uninitialized."""
import ctypes as C
import struct
import torch
from audit_half_native_row_grid59 import RowOwner
assert not torch.cuda.is_initialized();m,n,k=32,40,24
class Extension:
 def get_handles(self,graph):return 100,200
class FakeAPI:
 def __init__(self):self.lib=self;self.calls=0
 def check(self,status,name):assert status==0
 def cuGraphExecKernelNodeSetParams(self,exe,node,params):self.calls+=1;return 0
 def read(self,graph,tensors):
  x,w,y=(tensors[n]for n in('input','weight','output'))
  return dict(nodes=[dict(type=0,node=hex(100),function=hex(200),name='owned_CPU_fixture',grid=[2,4,1],block=[128,1,1],shared_bytes=0,matched_tensors=['input','weight','output'],arguments=[dict(index=0,bytes_hex=struct.pack('<QQQ',x.data_ptr(),w.data_ptr(),y.data_ptr()).hex())])])
passed=0
for rank in(0,1):
 x=torch.empty((2*m,k),dtype=torch.float16)[:m];w=torch.empty((n,k),dtype=torch.float16);y=torch.empty((2*m,n),dtype=torch.float16)[:m];api=FakeAPI()
 owner=RowOwner(None,x,w,y,Extension(),api,rank);owner.enable(1)
 assert owner.modified.params.grid_y==2 and owner.modified.params.grid_x==2
 raw=bytes(owner.modified.buffers[0]);a,b,c=struct.unpack('<QQQ',raw)
 assert a==x.data_ptr()+rank*(m//2)*k*2 and b==w.data_ptr()and c==y.data_ptr()+rank*(m//2)*n*2
 owner.close();assert api.calls==2;passed+=1
 for invalid_x,invalid_y in[(torch.empty((m,k),dtype=torch.float16),y),(x,torch.empty((m,n),dtype=torch.float16))]:
  try:RowOwner(None,invalid_x,w,invalid_y,Extension(),FakeAPI(),rank)
  except AssertionError:passed+=1
  else:raise AssertionError('Unpadded operand accepted')
assert not torch.cuda.is_initialized()
print('HALF_NATIVE_ROW_GRID59_CPU passed=true cases='+str(passed)+' unpadded_inputs_and_outputs_rejected=true weight_unchanged=true no_GPU=true')
