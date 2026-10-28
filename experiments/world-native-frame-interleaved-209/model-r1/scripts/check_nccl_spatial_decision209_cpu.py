"""Check exact pointer offsets, unchanged weights, grids and grouped fallbacks."""
from collections import Counter
import json
from types import SimpleNamespace
from unittest.mock import patch
import torch
import nccl_spatial_decision209_mode as mode


def main():
    checks=0;m=256;c=k=n=64
    for rank in (0,1):
        scope=dict(M=m,C=c,shaders=Counter())
        def run(name,values):
            jit=SimpleNamespace(arg_names=list(values),fn=SimpleNamespace(__name__=name,__module__='unifolm_wma.utils.'+{'_geglu':'triton_primary_narrow_geglu','_primary':'triton_primary_narrow_linear'}.get(name,'unused_fixture')))
            def parent(function,*args,**kwargs):
                assert function is jit
                return dict(zip(jit.arg_names,args)),kwargs
            with patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
                return mode.dispatch(jit,tuple(values.values()),dict(grid=(999,),warmup=False,enable_fp_fusion=False),scope,rank,parent)
        for name in ('_presplit_two_product_gemm','_packed_attention_output_gemm','_half_compensated_gemm','_two_product_gemm','_packed_two_product_gemm_residual','_zero_low_gemm_residual','_zero_low_gemm','_zero_low_gemm_persistent'):
            a=dict(M=m,N=n,K=k,BM=64,BN=64,BK=64,GROUP_M=32,OUTPUT_GROUPS=1,THREE=True)
            for slot in ('X','XL','XH','Residual','Output','Y'):a[slot]=torch.empty((m,k))
            a['WH']=torch.empty((n,k));a['LowCount']=torch.zeros((),dtype=torch.int64)
            b,opts=run(name,a);assert b['M']==m//2 and opts['grid']==(2,)
            assert b['WH'] is a['WH'] and b['LowCount'] is a['LowCount']
            inputs={'_presplit_two_product_gemm':('X','XL'),'_packed_attention_output_gemm':('XH','XL'),'_half_compensated_gemm':('X',),'_two_product_gemm':('X',)}.get(name,('XH','XL'))
            for slot in inputs:
                assert b[slot].shape==(m//2,k) and b[slot].data_ptr()==a[slot].data_ptr()+rank*(m//2)*k*4
            checks+=1
        for name in ('_zero_low_gemm','_zero_low_gemm_persistent'):
            a=dict(M=m,N=3*c,K=k,BM=64,BN=64,OUTPUT_GROUPS=3,THREE=False,XH=torch.empty((m,k)),Y=torch.empty((3,m,c)))
            b,opts=run(name,a);assert b['M']==m and b['Y'] is a['Y'] and opts['grid']==(999,);checks+=1
        for name in ('_geglu_subtile32','_both_geglu_subtile32'):
            a=dict(M=m,C=c*4,K=k,BM=64,BN=64,BK=64,DEBUG=False)
            for slot in ('XH','XL'):a[slot]=torch.empty((m,k),dtype=torch.float16)
            for slot in ('YH','YL','V','G'):a[slot]=torch.empty((m,c*4),dtype=torch.float16)
            b,opts=run(name,a);assert b['M']==m//2 and opts['grid']==(8,)
            assert b['V'] is a['V'] and b['G'] is a['G']
            assert b['YH'].data_ptr()==a['YH'].data_ptr()+rank*(m//2)*c*4*2;checks+=1
        a=dict(M=m,C=c,BM=64,Count=torch.zeros((),dtype=torch.int64))
        for slot in ('XH','XL','Q','KH','KL','VH','VL'):a[slot]=torch.empty((m,c))
        b,opts=run('_qkv_pack',a);assert b['M']==m//2 and b['Count'] is a['Count'] and opts['grid']==(6,);checks+=1
        a=dict(M=m//16,N=m//16,H=1,BM=64,BN=64)
        for slot in ('Q','KH','KL','VH','VL','OH','OL'):a[slot]=torch.empty((16,m//16,1,64))
        b,opts=run('_attention_hidden_parts_vector',a);assert b['M']==a['M'] and opts['grid']==(1,8)
        assert b['Q'].shape[0]==8 and b['Q'].data_ptr()==a['Q'].data_ptr()+rank*8*(m//16)*c*4;checks+=1
        a=dict(M=m,N=n,K=k,BM=64,BN=64,T=1,H=4,W=4,TEMPORAL=False,X=torch.empty((16,16,k)),Output=torch.empty((16,n,4,4)),Residual=torch.empty((n,16,4,4)).permute(1,0,2,3))
        b,opts=run('_out_projection_residual',a);assert b['M']==m//2 and b['Output'].shape[0]==8
        assert b['Residual'].stride()==a['Residual'].stride() and b['Residual'].data_ptr()==a['Residual'].data_ptr()+rank*8*a['Residual'].stride(0)*4;checks+=1
        for name in ('_residual_layer_norm','_residual_norm_half_pack'):
            a=dict(CHANNELS=c)
            for slot in ('Left','Right','Residual','Normalized','High','Low'):a[slot]=torch.empty((m,c))
            a['Weight']=torch.empty(c);a['Bias']=torch.empty(c)
            b,opts=run(name,a);assert opts['grid']==(m//2,) and b['Weight'] is a['Weight'] and b['High'].shape==(m//2,c);checks+=1
        a=dict(N=m*c,BLOCK=1024)
        for slot in ('Base','Image','State','Action','Output'):a[slot]=torch.empty((m,c))
        b,opts=run('_mix',a);assert b['N']==m*c//2 and opts['grid']==(8,);checks+=1
        a=dict(SIZE=m*c,SCALE=1.,BLOCK=1024,Weight=torch.empty((m,c)),High=torch.empty((m,c),dtype=torch.float16),Low=torch.empty((m,c),dtype=torch.float16))
        b,opts=run('_pack_half_weight',a);assert b['SIZE']==m*c//2 and opts['grid']==(8,);checks+=1
        for name in ('inline_zero_low_gemm','_primary'):
            a=dict(M=m,N=n,K=k,BM=64,BN=64,BK=64,GROUP_M=32,X=torch.empty((m,k)),Output=torch.empty((m,n)),Y=torch.empty((m,n)),LowCount=torch.zeros((),dtype=torch.int64),Weight=torch.empty((n,k)))
            b,opts=run(name,a);assert b['M']==m//2 and opts['grid']==(2,)
            assert b['Weight'] is a['Weight'] and b['LowCount'] is a['LowCount']
            assert b['X'].data_ptr()==a['X'].data_ptr()+rank*(m//2)*k*4;checks+=1
        for name in ('_geglu','_primary_geglu_hidden_parts','_primary_geglu_hidden_high'):
            a=dict(M=m,C=c*4,K=k,BM=64,BN=64,DEBUG=False,XH=torch.empty((m,k),dtype=torch.float16),XL=torch.empty((m,k),dtype=torch.float16))
            for slot in ('YH','YL','Y','DebugValue','DebugGate'):a[slot]=torch.empty((m,c*4),dtype=torch.float16)
            b,opts=run(name,a);assert b['M']==m//2 and opts['grid']==(8,)
            output='YH' if name=='_primary_geglu_hidden_parts' else 'Y'
            assert b[output].data_ptr()==a[output].data_ptr()+rank*(m//2)*c*4*2
            assert b['DebugValue'] is a['DebugValue'] and b['DebugGate'] is a['DebugGate'];checks+=1
        a=dict(M=m,N=c,K=k,BM=128,BN=128,X=torch.empty((m,k)),W=torch.empty((c*3,k),dtype=torch.float16))
        for slot in ('Q','KH','KL','VH','VL'):a[slot]=torch.empty((m,c))
        b,opts=run('_primary_qkv_pack',a);assert b['M']==m//2 and b['W'] is a['W'] and opts['grid']==(2,);checks+=1
        a=dict(M=m//16,N=m//16,H=1,BM=64,BN=64)
        for slot in ('Q','KH','KL','VH','VL','O'):
            a[slot]=torch.empty((16,m//16,1,64) if slot in ('Q','O') else (m,64))
        b,opts=run('_prepacked_attention',a);assert b['M']==a['M'] and opts['grid']==(1,8) and b['O'].shape[0]==8;checks+=1
        # Equal-width context rows are not the spatial activation M.
        a=dict(M=16*77,N=n,K=k,X=torch.empty((16*77,k)))
        b,opts=run('_two_product_gemm',a);assert b['M']==a['M'] and b['X'] is a['X'];checks+=1
    mode.uninstall()
    assert not torch.cuda.is_initialized()
    print('NCCL_SPATIAL_DECISION_CPU_CHECK '+json.dumps(dict(passed=True,checks=checks,no_GPU_initialized=True)))


if __name__=='__main__':main()
