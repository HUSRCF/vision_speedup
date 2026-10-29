"""Check selected FFN domains and CPU gather reconstruction without CUDA."""
import inspect,json,textwrap
from pathlib import Path
import torch
from nonv2_native_ffn_grid59_mode import choose,gather_rows
from unifolm_wma.utils.cuda_graph import DenoiserGraph
assert not torch.cuda.is_initialized()
source=textwrap.dedent(inspect.getsource(DenoiserGraph.capture));assert source.count('    graph = torch.cuda.CUDAGraph()')==1
q=json.loads((Path(__file__).parent.parent/'qualified-shapes.json').read_text());sets=[{tuple(r['shape'])for r in rank if r['supported']}for rank in q['ranks']];assert sets[0]==sets[1]and len(sets[0])==9
for shape in[(16,2560,320),(2560,16,320),(16,640,640),(640,16,640),(16,160,1280),(160,16,1280),(2560,16,512)]:assert choose(shape,sets[0])
for shape in[(16,40,1280),(40,16,1280)]:assert not choose(shape,sets[0])
for shape in[(16,4,8),(8,16,4)]:
 x=torch.arange(torch.tensor(shape).prod()).reshape(shape).half();flat=x.view(-1,x.shape[-1]);half=flat.shape[0]//2
 for rank in(0,1):
  def gather(output,local):
   assert torch.equal(local,flat[rank*half:(rank+1)*half]);output.copy_(flat)
  assert torch.equal(gather_rows(x,rank,gather),x)
assert not torch.cuda.is_initialized()
print('NONV2_NATIVE_FFN_GRID59_CPU passed=true qualified9_shapes=true small640row_producer_rejected=true full16_and_complete_temporal_rows=true no_GPU=true')
