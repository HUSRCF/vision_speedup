"""Reuse an already validated Winograd kernel for aligned output-channel halves.

All original constexprs, CUBIN, K order and batch strides remain unchanged.
Only validated owner weight/output starts and the row tile grid change.
No native workspace is dereferenced or wrapped as a guessed Torch tensor.
"""
import struct
import torch
import triton
from unifolm_wma.utils.native_graph_weight_cache_probe import OwnedParams


def domain(geometry,config,rank):
    co,cols,k=geometry;bm,bn,bk,warps,stages=config
    assert rank in(0,1)and co%2==0 and co//2%bm==0
    half=co//2
    assert (rank*half*k*2)%16==0 and (rank*half*cols*4)%16==0
    return dict(half=half,row_start=rank*half,grid=[half//bm*triton.cdiv(cols,bn),36,1])


def parameters(owner,geometry,config,rank):
    d=domain(geometry,config,rank);co,cols,k=geometry;bm,bn,bk,warps,stages=config
    original=owner.replacement;packed=hasattr(owner,'wh');count=4 if packed else 3
    assert len(original.buffers)==count and all(len(b)==8 for b in original.buffers)
    p=[struct.unpack('<Q',bytes(b))[0]for b in original.buffers]
    if packed:
        assert p[:2]==[owner.wh.data_ptr(),owner.wl.data_ptr()]
        for i in (0,1):p[i]+=d['row_start']*k*2
    else:
        assert p[0]==owner.cache.cache.data_ptr();p[0]+=d['row_start']*k*4
    p[-1]+=d['row_start']*cols*4
    old=original.params
    assert (old.grid_x,old.grid_y,old.grid_z)==(2*d['grid'][0],36,1)
    assert (old.block_x,old.block_y,old.block_z)==(32*warps,1,1)
    return OwnedParams(dict(function=hex(old.func),grid=d['grid'],block=[old.block_x,old.block_y,old.block_z],shared_bytes=old.shared,
        arguments=[dict(index=i,offset=i*8,size=8,bytes_hex=struct.pack('<Q',v).hex())for i,v in enumerate(p)]))


def gather_channels(value,rank):
    import torch.distributed as dist
    assert value.ndim==4 and value.shape[0]==16 and value.shape[1]%2==0
    n,c,h,w=value.shape;half=c//2
    local=value[:,rank*half:(rank+1)*half].contiguous()
    packed=torch.empty((2*n,half,h,w),device=value.device,dtype=value.dtype)
    dist.all_gather_into_tensor(packed,local)
    # Preserve the exact output stride seen by subsequent cuDNN dispatch.
    output=torch.empty_like(value,memory_format=torch.preserve_format)
    source=packed.view(2,n,half,h,w).permute(1,0,2,3,4)
    output[:,0:half].copy_(source[:,0]);output[:,half:].copy_(source[:,1])
    assert output.stride()==value.stride()
    return output
