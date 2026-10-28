"""Pinned Triton2.3.1 direct CUfunction parameters, validated against captures."""
import ctypes as C
import hashlib
import struct
import torch
import triton
from .native_graph_weight_cache_probe import OwnedParams
from .native_graph_world_filter_cache_probe import GEMM
from .native_winograd_sgemm_swap import NativeSgemmSwap


def parameters(api,kernel,pointers,geometry,config):
    assert torch.__version__=='2.3.1+cu121' and triton.__version__=='2.3.1'
    co,m,ci=geometry;bm,bn,bk,warps,stages=config
    kernel._init_handles()
    assert kernel.num_warps==warps and kernel.num_ctas==1 and tuple(kernel.cluster_dims)==(1,1,1)
    assert not kernel.metadata.get('tensormaps_info')
    for index in range(4):
        offset,size=C.c_size_t(),C.c_size_t()
        status=api.lib.cuFuncGetParamInfo(C.c_void_p(kernel.function),index,C.byref(offset),C.byref(size))
        if index==3:assert status==1,'Unexpected extra kernel argument';continue
        api.check(status,'direct Triton parameter metadata')
        assert (offset.value,size.value)==(index*8,8)
    node=dict(function=hex(kernel.function),grid=[triton.cdiv(co,bm)*triton.cdiv(m,bn),36,1],
        block=[32*warps,1,1],shared_bytes=kernel.shared,
        arguments=[dict(index=i,offset=8*i,size=8,bytes_hex=struct.pack('<Q',ptr).hex()) for i,ptr in enumerate(pointers)])
    return OwnedParams(node)


def witness(candidate_graph,kernel,pointers,geometry,config,extension,api):
    raw,_=extension.get_handles(candidate_graph)
    nodes=api.read(raw,{})['nodes'];assert len(nodes)==1 and nodes[0]['type']==0
    observed=OwnedParams(nodes[0]);direct=parameters(api,kernel,pointers,geometry,config)
    fields=['func','grid_x','grid_y','grid_z','block_x','block_y','block_z','shared']
    assert all(getattr(observed.params,n)==getattr(direct.params,n) for n in fields)
    assert len(observed.buffers)==len(direct.buffers)==3
    assert [bytes(b) for b in observed.buffers]==[bytes(b) for b in direct.buffers]
    return dict(captured_function_grid_block_shared_and_argument_bytes_identical=True,
        bounded_driver_parameter_metadata=True,cubin_sha256=hashlib.sha256(kernel.kernel).hexdigest())


class DirectSgemmSwap(NativeSgemmSwap):
    def __init__(self,graph,native_node,kernel,geometry,config,extension,api):
        self.graph,self.candidate_graph,self.api=graph,None,api
        self.raw,self.executable=extension.get_handles(graph)
        assert native_node['name']==GEMM and [a['size'] for a in native_node['arguments']]==[176]
        payload=bytes.fromhex(native_node['arguments'][0]['bytes_hex'])
        a,b,c=struct.unpack_from('<3Q',payload)
        assert struct.unpack_from('<I',payload,152)[0]==0x3f800000
        assert struct.unpack_from('<Q',payload,160)[0]==0 and struct.unpack_from('<Q',payload,168)[0]==c
        co,m,ci=geometry
        assert b-a==36*m*ci*4 and c-b==36*co*ci*4
        assert struct.unpack_from('<3Q',payload,56)==(m*ci,ci*co,m*co)
        self.original=OwnedParams(native_node)
        self.replacement=parameters(api,kernel,(b,a,c),geometry,config)
        self.kernel=kernel
        self.node=int(native_node['node'],16);self.enabled=False;self.closed=False
        self.proof=dict(original_native_input_filter_output_transforms=True,numerical_SGEMM_only=True,
            candidate_real_capture_params=False,direct_params_driver_metadata=True,
            candidate_pointer_order=['native_filter_B','native_input_A','native_output_C'])
