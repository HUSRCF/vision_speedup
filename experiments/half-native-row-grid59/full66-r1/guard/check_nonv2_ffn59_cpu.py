"""Check selected FFN domains and CPU gather reconstruction without CUDA."""
import inspect,json,textwrap
from pathlib import Path
import torch
from unifolm_wma.utils.nonv2_native_ffn_grid59_mode import choose,gather_rows
from unifolm_wma.utils.cuda_graph import DenoiserGraph
assert not torch.cuda.is_initialized()
source=textwrap.dedent(inspect.getsource(DenoiserGraph.capture));assert source.count('    graph = torch.cuda.CUDAGraph()')==1
q=json.loads((Path(__import__('unifolm_wma.utils.nonv2_native_ffn_grid59_mode',fromlist=['__file__']).__file__).parent/'qualified-half-native-ffn64.json').read_text());sets=[{tuple(r['shape'])for r in rank if r['supported']}for rank in q['ranks']];assert sets[0]==sets[1]and len(sets[0])==9
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

"""Validate original GEGLU row addressing and unchanged whole-tile masks."""
import torch
from unifolm_wma.utils.nonv2_native_ffn_grid59_geglu_mode import local_arguments
assert not torch.cuda.is_initialized();x=torch.arange(8192).reshape(32,256).half();y=torch.empty((32,128),dtype=torch.float16);n=y.numel();cases=0
for rank in(0,1):
 a,b=local_arguments(x,y,n,128,1024,rank)
 assert torch.equal(a,x[rank*16:(rank+1)*16].flatten())and b.data_ptr()==y[rank*16:(rank+1)*16].data_ptr();cases+=1
for m,c in[(40960,320),(10240,640),(2560,1280),(40960,512)]:
 n=m*4*c;assert n//2%1024==0 and n//2%(4*c)==0
 last=(n//2//1024-1)*1024+torch.arange(1024)
 assert bool((last<n).all())and int(last[-1])==n//2-1;cases+=1
assert not torch.cuda.is_initialized()
print('NONV2_NATIVE_FFN_GEGLU_CPU passed=true cases='+str(cases)+' exact_row_domains_and_original_masks=true no_GPU=true')
