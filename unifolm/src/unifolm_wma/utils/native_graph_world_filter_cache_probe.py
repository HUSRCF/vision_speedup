"""Diagnostic native Winograd filter caching; preserve every arithmetic kernel."""
import ctypes as C
import struct
import torch
from .native_graph_weight_cache_probe import NativeWeightCache, OwnedParams


FILTER = '_ZN5cudnn17winograd_nonfused24winogradForwardFilter4x4IffEEvNS0_20WinogradFilterParamsIT_T0_EE'
GEMM = 'ampere_sgemm_128x128_nn'


def eligible_world_plan(inventory, weight):
    if (torch.is_grad_enabled() or weight.dtype != torch.float32 or not weight.is_cuda
            or not weight.is_contiguous() or weight.ndim != 4
            or tuple(weight.shape[2:]) != (3,3) or weight.shape[0]%32 or weight.shape[1]%32):
        return None
    nodes = {n['node']:n for n in inventory['nodes']}
    readers = [n for n in nodes.values() if n.get('name') == FILTER
               and n.get('matched_tensors') == ['weight']]
    if len(readers) != 1:
        return None
    transform = readers[0]
    if [a['size'] for a in transform['arguments']] != [40]:
        return None
    payload = bytes.fromhex(transform['arguments'][0]['bytes_hex'])
    co, ci = weight.shape[:2]
    if struct.unpack_from('<4I',payload) != (co,ci,co*ci,0):
        return None
    original_input, old_output, flags = struct.unpack_from('<3Q',payload,16)
    if original_input != weight.data_ptr() or flags != 0:
        return None
    successors = [nodes[b] for a,b in inventory['edges'] if a == transform['node']]
    if len(successors) != 1:
        return None
    consumer = successors[0]
    if (consumer.get('name') != GEMM or [a['size'] for a in consumer.get('arguments',[])] != [176]
            or consumer['grid'][2] != 36):
        return None
    gemm = bytes.fromhex(consumer['arguments'][0]['bytes_hex'])
    # Independent evidence of the exact filter allocation span: 36 batched
    # matrices, native B batch stride CO*CI, then the next output workspace.
    numel = 36*co*ci
    if (struct.unpack_from('<Q',gemm,8)[0] != old_output
            or struct.unpack_from('<Q',gemm,64)[0] != co*ci
            or struct.unpack_from('<Q',gemm,16)[0] != old_output+numel*4):
        return None
    return transform, consumer, old_output, numel


class NativeWorldFilterCache(NativeWeightCache):
    def __init__(self, graph, weight, extension, api, validated_plan=None):
        assert not torch.cuda.is_current_stream_capturing()
        self.graph, self.weight, self.api = graph, weight, api
        self.raw, self.executable = extension.get_handles(graph)
        plan = validated_plan or eligible_world_plan(api.read(self.raw,dict(weight=weight)),weight)
        if plan is None:
            raise ValueError('No validated native Winograd filter/SGEMM pair')
        transform, consumer, self.old_output, numel = plan
        self.transform_node = int(transform['node'],16)
        self.consumer_node = int(consumer['node'],16)
        self.guard_elements = 4096
        self.backing = torch.full((numel+2*self.guard_elements,),90909.,device=weight.device,dtype=torch.float32)
        self.cache = self.backing[self.guard_elements:-self.guard_elements]
        self.cache.fill_(float('nan'))
        self.initial_shape, self.initial_ptr = tuple(weight.shape), weight.data_ptr()
        self.version, self.refreshes = None, 0
        self.enabled = self.closed = False
        self.transform = OwnedParams(transform,{(0,24):(self.old_output,self.cache.data_ptr())})
        self.original_consumer = OwnedParams(consumer)
        self.cached_consumer = OwnedParams(consumer,{(0,8):(self.old_output,self.cache.data_ptr())})
        before, after = bytes(self.original_consumer.buffers[0]),bytes(self.cached_consumer.buffers[0])
        assert before[:8] == after[:8] and before[16:] == after[16:]
        self.refresh_graph, self.refresh_exec = C.c_void_p(), C.c_void_p()
        api.check(api.lib.cuGraphCreate(C.byref(self.refresh_graph),0),'create native filter refresh graph')
        node = C.c_void_p()
        api.check(api.lib.cuGraphAddKernelNode(C.byref(node),self.refresh_graph,None,0,
                  C.byref(self.transform.params)),'clone original native filter transform')
        api.check(api.lib.cuGraphInstantiateWithFlags(C.byref(self.refresh_exec),self.refresh_graph,0),
                  'instantiate native filter refresh graph')
        self.refresh_if_needed()
        self.proof = dict(transform=transform['name'],consumer=consumer['name'],
            transform_output_argument=0,transform_output_byte=24,consumer_filter_byte=8,
            cached_bytes=numel*4,span_proof='36 native SGEMM batches * CO*CI stride; next output starts exactly at span end',
            original_native_transform_cloned=True,only_consumer_filter_pointer_changed=True)

    def verify_cache(self):
        assert torch.isfinite(self.cache).all()
        self.verify_canaries()

    def poison_old_workspace(self):
        assert self.enabled
        self.api.check(self.api.lib.cuMemsetD32Async(self.old_output,0x7fc00000,self.cache.numel(),
            C.c_void_p(torch.cuda.current_stream(self.weight.device).cuda_stream)),
            'poison original full transformed filter span')
