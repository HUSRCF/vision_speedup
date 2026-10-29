"""Check global-frame masks, complete temporal rows and native-pair grouping."""
import torch
from unifolm_wma.utils.nccl_spatial_decision209_mode import half_rows
from unifolm_wma.utils.nccl_temporal_block221_mode import row_view,grouped_route
from unifolm_wma.utils.nccl_initial_ffn220_mode import row_values
from unifolm_wma.utils.nccl_initial_block222_mode import DeferredInitialGather
from unifolm_wma.utils.nccl_native_attention219_mode import local_native
from unifolm_wma.utils.four_way226_groups import degree

assert not torch.cuda.is_initialized();cases=0
for m in (640,2560,10240,40960):
    x=torch.arange(m*8).reshape(m,8)
    for rank in range(4):
        expected=x[rank*(m//4):(rank+1)*(m//4)]
        assert torch.equal(half_rows(x,m,8,rank),expected)
        assert torch.equal(row_view(x,m,8,rank),expected)
        assert expected.shape[0]%16==0;cases+=1
for rank in range(4):
    x=torch.arange(64*64).reshape(64,64)
    values=dict(M=64,N=192,K=64,OUTPUT_GROUPS=3,XH=x,XL=x,Y=torch.empty(3,64,64),BM=64,BN=64)
    routed,grid=grouped_route(values,rank,torch.empty(3,16,64))
    assert routed['M']==16 and routed['Y'].shape==(3,16,64) and grid==(3,)
    assert torch.equal(routed['XH'],x[rank*16:(rank+1)*16]);cases+=1
    q=torch.arange(16*3*2*4,dtype=torch.float32).reshape(16,3,2,4);bias=torch.arange(16*2*3*3).reshape(16,2,3,3)
    for mask in (None,bias,bias[:1]):
        def parent(a,b,c,attn_bias,**kwargs):
            assert torch.equal(a,q[rank*4:(rank+1)*4])
            if mask is None:assert attn_bias is None
            else:assert torch.equal(attn_bias,mask[rank*4:(rank+1)*4] if mask.shape[0]==16 else mask)
            return a.clone()
        result,report=local_native(parent,q,q,q,mask,rank=rank)
        assert torch.equal(result[rank*4:(rank+1)*4],q[rank*4:(rank+1)*4]) and report['local_batch']==4;cases+=1
    output=torch.empty(40960,512);local=output[rank*10240:(rank+1)*10240]
    seen=[]
    def gather(target,value):
        assert value.shape==(10240,512) and value.data_ptr()==target[rank*10240:(rank+1)*10240].data_ptr();seen.append(True)
    deferred=DeferredInitialGather(rank,gather);deferred.record(output,local);deferred.finish(torch.empty_like(output));assert seen==[True]
    assert deferred.pending is None and deferred.completed==1;cases+=1
    producer=dict(M=40960,C=2048,K=512,BM=64,BN=64,GROUP=32,DEBUG=False,X=torch.empty(40960,512),YH=torch.empty(40960,2048),YL=torch.empty(40960,2048))
    routed,grid=row_values(producer,'producer',rank);assert routed['M']==10240 and routed['X'].shape==(10240,512) and grid==(5120,);cases+=1
assert degree(torch.empty(16,8),torch.empty(4,8))==4 and degree(torch.empty(16,8),torch.empty(8,8))==2
try:degree(torch.empty(12,8),torch.empty(4,8));raise AssertionError('Expected invalid degree rejection')
except AssertionError as error:
    assert str(error)!='Expected invalid degree rejection'
assert not torch.cuda.is_initialized()
print('FOUR_WAY226_CPU passed=true cases='+str(cases)+' global_mask_slices=true complete16T=true native_pair_degree2=true no_GPU=true')
