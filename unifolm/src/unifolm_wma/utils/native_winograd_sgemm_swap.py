"""Diagnostic numerical replacement of SGEMM, retaining native transforms."""
import ctypes as C
import struct
import torch
from .native_graph_weight_cache_probe import OwnedParams
from .native_graph_world_filter_cache_probe import GEMM


class NativeSgemmSwap:
    def __init__(self,graph,native_node,candidate_graph,expected,extension,api):
        self.graph,self.candidate_graph,self.api=graph,candidate_graph,api
        self.raw,self.executable=extension.get_handles(graph)
        assert native_node['name']==GEMM and [a['size'] for a in native_node['arguments']]==[176]
        native=bytes.fromhex(native_node['arguments'][0]['bytes_hex'])
        a,b,c=struct.unpack_from('<3Q',native)
        assert struct.unpack_from('<I',native,152)[0]==0x3f800000
        assert struct.unpack_from('<Q',native,160)[0]==0 and struct.unpack_from('<Q',native,168)[0]==c
        raw,_=extension.get_handles(candidate_graph)
        nodes=api.read(raw,{})['nodes']
        # Copy a real captured Triton node. Its CUfunction, grid, block,
        # shared memory and metadata-backed argument order are observed,
        # rather than reconstructed from a compiler implementation detail.
        assert len(nodes)==1 and nodes[0]['type']==0
        node=nodes[0];args=node['arguments']
        assert [x['size'] for x in args]==[8,8,8] and [x['offset'] for x in args]==[0,8,16]
        assert [int.from_bytes(bytes.fromhex(x['bytes_hex']),'little') for x in args]==list(expected)
        self.original=OwnedParams(native_node)
        self.replacement=OwnedParams(node,{(i,0):(old,new) for i,(old,new) in enumerate(zip(expected,(b,a,c)))})
        self.node=int(native_node['node'],16);self.enabled=False;self.closed=False
        self.proof=dict(original_native_input_filter_output_transforms=True,
            numerical_SGEMM_only=True,candidate_real_capture_params=True,
            candidate_pointer_order=['native_filter_B','native_input_A','native_output_C'])

    def set_enabled(self,enabled):
        assert not self.closed
        p=self.replacement if enabled else self.original
        self.api.check(self.api.lib.cuGraphExecKernelNodeSetParams(self.executable,C.c_void_p(self.node),
            C.byref(p.params)),'set numerical SGEMM-only executable params')
        self.enabled=enabled

    def close(self):
        if self.closed:return
        torch.cuda.synchronize();self.set_enabled(False);self.closed=True
