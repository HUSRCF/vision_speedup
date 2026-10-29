"""Check spatial/time FFN sample ownership and gather shape on real CPU views."""
import json
import torch
from nonv2_ffn59_mode import row_inference
assert not torch.cuda.is_initialized();cases=0
shapes=[(16,2560,320),(16,640,640),(16,160,1280),(16,40,1280),(2560,16,320),(640,16,640),(160,16,1280),(40,16,1280),(2560,16,512)]
for dtype in (torch.float16,torch.float32):
 for shape in shapes:
  value=(torch.arange(shape[0],dtype=torch.float32)[:,None,None]%17).expand(shape).contiguous().to(dtype)
  def original(x):return (x*.5)+.25
  expected=original(value)
  for rank in (0,1):
   observations=[];calls=[];part=shape[0]//2
   def forward(x):
    assert x.shape==(part,*shape[1:]) and x.is_contiguous()
    assert x.data_ptr()==value[rank*part].data_ptr();observations.append(1);return original(x)
   def gather(output,local):
    assert output.shape==shape and local.shape==(part,*shape[1:])
    assert torch.equal(local,expected[rank*part:(rank+1)*part]);output.copy_(expected);calls.append(1)
   output=row_inference(value,rank,forward,gather)
   assert torch.equal(output,expected) and len(calls)==len(observations)==1;cases+=1
 del value,expected,output
assert not torch.cuda.is_initialized()
print('NONV2_FFN59_CPU '+json.dumps(dict(passed=True,cases=cases,no_GPU_initialized=True,sample_axis_only=True,one_gather=True,not_GPU_numerical_qualification=True)))
