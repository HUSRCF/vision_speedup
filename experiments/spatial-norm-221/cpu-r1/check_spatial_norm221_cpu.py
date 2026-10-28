"""Check real CPU view offsets, all scalar retention, temporal rejection."""
import json
import torch
from spatial_norm221_mode import route_norm
assert not torch.cuda.is_initialized();cases=0;rejected=0
for c,h,w in ((320,40,64),(640,20,32),(1280,10,16),(1280,5,8)):
 area=h*w;m=16*area;size=c//32*area;parts=(size+4095)//4096
 for contiguous in (False,True):
  x=torch.empty((16,c,h,w)) if contiguous else torch.empty((c,16,h,w)).transpose(0,1)
  out=torch.empty((16,c,h,w));stats=torch.empty((512,parts,2));moments=torch.empty((512,2));weight=torch.empty(c);bias=torch.empty(c)
  high=torch.empty((m,c),dtype=torch.float16);low=torch.empty_like(high);debug=out.view(m,c)
  base=dict(X=x,Stats=stats,Moments=moments,Output=out,Weight=weight,Bias=bias,SIZE=size,PARTS=parts,GROUPS=32,CHANNELS=c,FRAMES=1,HEIGHT=h,WIDTH=w,S0=x.stride(0),S1=x.stride(1),ST=0,SH=x.stride(2),SW=x.stride(3),DEBUG=False)
  for rank in (0,1):
   for name,grid in (('_partial',(512,parts)),('_finish_stats',(512,)),('_apply',(512,(size+1023)//1024)),('_serial_norm',(512,)),('_apply_layout_pack',(m//32,(c+31)//32))):
    if name=='_serial_norm' and not contiguous:continue
    values=dict(base)
    if name=='_apply_layout_pack':values.update(M=m,C=c,G=32,T=1,H=h,W=w,TEMPORAL=False,DEBUG=True,RM=32,RC=32,PH=high,PL=low,Debug=debug)
    routed,newgrid=route_norm(name,values,grid,dict(M=m,C=c),rank)
    assert {k:v for k,v in routed.items() if not isinstance(v,torch.Tensor)}=={k:v for k,v in values.items() if not isinstance(v,torch.Tensor)}
    assert routed['Weight'] is weight and routed['Bias'] is bias and newgrid[0]==grid[0]//2 and newgrid[1:]==grid[1:]
    if name!='_finish_stats':assert routed['X'].storage_offset()==rank*8*x.stride(0)
    if name in ('_partial','_finish_stats'):assert routed['Stats'].storage_offset()==rank*256*parts*2
    if name in ('_finish_stats','_apply','_apply_layout_pack'):assert routed['Moments'].storage_offset()==rank*256*2
    if name in ('_apply','_serial_norm'):assert routed['Output'].storage_offset()==rank*8*c*area
    if name=='_apply_layout_pack':
     for slot in ('PH','PL','Debug'):assert routed[slot].storage_offset()==rank*8*c*area
    cases+=1
   temporal=dict(base,FRAMES=16)
   try:route_norm('_partial',temporal,(512,parts),dict(M=m,C=c),rank)
   except AssertionError:rejected+=1
   else:raise AssertionError('Temporal GroupNorm accepted')
 del x,out,stats,moments,weight,bias,high,low,debug,base,values,routed
assert not torch.cuda.is_initialized()
print('SPATIAL_NORM221_CPU '+json.dumps(dict(passed=True,cases=cases,temporal_rejections=rejected,no_GPU_initialized=True,scalar_dimensions_strides_unchanged=True,only_row_pointers_and_grid_routed=True)))
