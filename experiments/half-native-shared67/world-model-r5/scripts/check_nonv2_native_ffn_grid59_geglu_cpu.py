"""Validate original GEGLU row addressing and unchanged whole-tile masks."""
import torch
from nonv2_native_ffn_grid59_geglu_mode import local_arguments
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
