"""Check gather occurs once after residual, with original row ownership."""
import json
import torch
from initial_boundary221_mode import DeferredInitialGather
assert not torch.cuda.is_initialized();cases=0
for rank in (0,1):
 for sign in (-1.,0.,1.):
  raw=torch.full((40960,512),float('nan'));rows=slice(rank*20480,(rank+1)*20480)
  local=raw[rows];local.fill_(sign*.25);residual=torch.full_like(raw,float('nan'));residual[rows].fill_(.125)
  calls=[]
  def gather(output,owned):
   assert output.shape==(40960,512) and owned.shape==(20480,512)
   assert owned.data_ptr()==output[rows].data_ptr()
   assert torch.all(owned==sign*.25+.125) and torch.isfinite(owned).all()
   calls.append((output,owned));output.fill_(sign*.25+.125)
  deferred=DeferredInitialGather(rank,gather);deferred.record(raw,local);assert not calls
  try:deferred.record(raw,local)
  except AssertionError:pass
  else:raise AssertionError('second raw FFN gather accepted')
  result=raw+residual;assert not calls
  returned=deferred.finish(result);assert returned is result and len(calls)==1 and deferred.completed==1 and deferred.pending is None
  assert torch.all(result==sign*.25+.125)
  try:deferred.finish(result)
  except AssertionError:pass
  else:raise AssertionError('second final gather accepted')
  cases+=1
 del raw,local,residual,result,returned,calls,deferred
assert not torch.cuda.is_initialized()
print('INITIAL_BOUNDARY221_CPU '+json.dumps(dict(passed=True,cases=cases,gather_after_residual=True,one_collective=True,no_GPU_initialized=True)))
