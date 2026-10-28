"""Check actual pointer/view routing, fallback and immutable kernel options."""
import json,torch
from unifolm_wma.utils.nccl_initial_ffn220_mode import row_values

def main():
    assert not torch.cuda.is_initialized()
    rows=[]
    for rank in (0,1):
        for kind in ('producer','consumer'):
            values=dict(M=40960,K=512 if kind=='producer' else 2048,BM=64,BN=64 if kind=='producer' else 128,HAS_BIAS=True)
            if kind=='producer':
                values.update(C=2048,GROUP=32,DEBUG=False)
                buffers={'X':512,'YH':2048,'YL':2048}
            else:
                values.update(N=512,BK=32,GROUP_M=32,OUTPUT_GROUPS=1)
                buffers={'XH':2048,'XL':2048,'Output':512}
            for name,width in buffers.items():values[name]=torch.empty((40960,width))
            # Weight/bias/dummy operands must retain their original identity.
            values.update(WH=object(),WL=object(),Bias=object())
            if kind=='producer':values.update(V=values['YH'],G=values['YH'])
            out,grid=row_values(values,kind,rank)
            assert values['M']==40960 and out['M']==20480
            for name,width in buffers.items():
                assert out[name].shape==(20480,width) and out[name].is_contiguous()
                assert out[name].untyped_storage().data_ptr()==values[name].untyped_storage().data_ptr()
                assert out[name].storage_offset()==rank*20480*width
            for name in set(values)-set(buffers)-{'M'}:assert out[name] is values[name]
            assert grid==((20480//64)*(2048//64 if kind=='producer' else 512//128),)
            rows.append(dict(rank=rank,kind=kind,passed=True,exact_half_rows=True,all_nonrow_arguments_unchanged=True))
            del values,out
    assert not torch.cuda.is_initialized()
    print('INIT_FFN219_CPU '+json.dumps(dict(passed=True,cases=4,rows=rows,no_GPU_initialized=True,CUDA_numerics_and_model_performance_pending=True)))
if __name__=='__main__':main()
