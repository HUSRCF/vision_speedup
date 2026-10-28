"""Diagnostic-only caching of one proven cuDNN weight transpose in a graph.

Keep the original convolution kernel and every parameter byte except its
validated filter pointer. Workspace-backed transformed weights are replaced by
separate persistent storage, so later workspace reuse cannot corrupt the cache.
"""
import ctypes as C
import struct
import torch
from cuda_graph_readonly_inventory import Inventory, KernelParams


TRANSPOSE = '_ZN5cudnn3ops16nchwToNhwcKernelIfffLb0ELb1EL21cudnnKernelDataType_t2EEEvNS0_18nchw2nhwc_params_tIT1_EEPKT_PT0_'
CUTLASS = '_ZN19cutlass_cudnn_infer6KernelI67cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4_nhwc_align4EEvNT_6ParamsE'
XMMA = 'sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_'
XMMA_NONINDEXED = 'sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_'


class GraphAPI(Inventory):
    def __init__(self):
        super().__init__()
        specs = {
            'cuGraphCreate': [C.POINTER(C.c_void_p), C.c_uint],
            'cuGraphAddKernelNode': [C.POINTER(C.c_void_p), C.c_void_p,
                                     C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(KernelParams)],
            'cuGraphInstantiateWithFlags': [C.POINTER(C.c_void_p), C.c_void_p, C.c_ulonglong],
            'cuGraphExecKernelNodeSetParams': [C.c_void_p, C.c_void_p, C.POINTER(KernelParams)],
            'cuGraphNodeSetEnabled': [C.c_void_p, C.c_void_p, C.c_uint],
            'cuGraphNodeGetEnabled': [C.c_void_p, C.c_void_p, C.POINTER(C.c_uint)],
            'cuGraphLaunch': [C.c_void_p, C.c_void_p],
            'cuGraphExecDestroy': [C.c_void_p],
            'cuGraphDestroy': [C.c_void_p],
            'cuMemsetD32Async': [C.c_ulonglong, C.c_uint, C.c_size_t, C.c_void_p],
            'cuMemcpyDtoDAsync_v2': [C.c_ulonglong, C.c_ulonglong, C.c_size_t, C.c_void_p],
        }
        for name, types in specs.items():
            fn = getattr(self.lib, name)
            fn.argtypes = types
            fn.restype = C.c_int


class OwnedParams:
    def __init__(self, node, replacements=None):
        self.buffers = []
        replacements = replacements or {}
        for argument in node['arguments']:
            data = bytearray.fromhex(argument['bytes_hex'])
            for (index, byte), (old, new) in replacements.items():
                if index == argument['index']:
                    assert int.from_bytes(data[byte:byte + 8], 'little') == old
                    data[byte:byte + 8] = int(new).to_bytes(8, 'little')
            self.buffers.append(C.create_string_buffer(bytes(data), len(data)))
        self.pointers = (C.c_void_p * len(self.buffers))(*[C.addressof(b) for b in self.buffers])
        self.params = KernelParams(int(node['function'], 16), *node['grid'], *node['block'],
                                   node['shared_bytes'], self.pointers, None)


def eligible_plan(inventory, weight):
    if (torch.is_grad_enabled() or weight.dtype != torch.float32 or not weight.is_cuda
            or not weight.is_contiguous() or weight.ndim not in {4, 5}
            or weight.shape[0] % 32 or weight.shape[1] % 32):
        return None
    if tuple(weight.shape[2:]) not in {(3, 3), (3, 1, 1)}:
        return None
    nodes = {node['node']: node for node in inventory['nodes']}
    readers = [node for node in nodes.values() if node.get('name') == TRANSPOSE
               and node.get('matched_tensors') == ['weight']]
    if len(readers) != 1:
        return None
    transform = readers[0]
    if [a['size'] for a in transform['arguments']] != [44, 8, 8]:
        return None
    co, ci = weight.shape[:2]
    spatial = weight.numel() // (co * ci)
    expected = (co, ci, ci, ci * spatial, spatial, spatial,
                ci, ci * spatial, ci, 0x3f800000, 0)
    if struct.unpack('<11I', bytes.fromhex(transform['arguments'][0]['bytes_hex'])) != expected:
        return None
    pointer = lambda arg: int.from_bytes(bytes.fromhex(arg['bytes_hex']), 'little')
    if pointer(transform['arguments'][1]) != weight.data_ptr():
        return None
    old_output = pointer(transform['arguments'][2])
    successors = [nodes[b] for a, b in inventory['edges'] if a == transform['node']]
    if len(successors) != 1:
        return None
    consumer = successors[0]
    passthrough = []
    for _ in range(2):
        if consumer['type'] != 2:
            break
        passthrough.append(consumer['node'])
        successors = [nodes[b] for a,b in inventory['edges'] if a == consumer['node']]
        if len(successors) != 1:
            return None
        consumer = successors[0]
    name = consumer.get('name', '')
    if name.startswith((XMMA, XMMA_NONINDEXED)) and len(consumer['arguments']) == 1 and consumer['arguments'][0]['size'] in {784,1440}:
        byte = 8
    elif name == CUTLASS and len(consumer['arguments']) == 1 and consumer['arguments'][0]['size'] == 464:
        byte = 240
    else:
        return None
    payload = bytes.fromhex(consumer['arguments'][0]['bytes_hex'])
    if int.from_bytes(payload[byte:byte + 8], 'little') != old_output:
        return None
    return transform, consumer, byte, old_output, passthrough


class NativeWeightCache:
    def __init__(self, graph, weight, extension, api, validated_plan=None):
        assert not torch.cuda.is_current_stream_capturing()
        self.graph, self.weight, self.api = graph, weight, api
        self.raw, self.executable = extension.get_handles(graph)
        plan = validated_plan
        if plan is None:
            inventory = api.read(self.raw, dict(weight=weight))
            plan = eligible_plan(inventory, weight)
        if plan is None:
            raise ValueError('No validated standalone weight-transpose/consumer pair')
        transform, consumer, byte, old_output, passthrough = plan
        self.transform_node = int(transform['node'], 16)
        self.consumer_node = int(consumer['node'], 16)
        self.old_output = old_output
        self.guard_elements = 4096
        self.backing = torch.full((weight.numel()+2*self.guard_elements,), 90909., device=weight.device, dtype=torch.float32)
        self.cache = self.backing[self.guard_elements:-self.guard_elements]
        self.cache.fill_(float('nan'))
        self.initial_shape = tuple(weight.shape)
        self.initial_ptr = weight.data_ptr()
        self.version = None
        self.refreshes = 0
        self.enabled = False
        self.closed = False
        self.transform = OwnedParams(transform, {(2, 0): (old_output, self.cache.data_ptr())})
        self.original_consumer = OwnedParams(consumer)
        self.cached_consumer = OwnedParams(consumer, {(0, byte): (old_output, self.cache.data_ptr())})
        # The only differing bytes in the consumer are the one native filter
        # pointer. In particular no arithmetic, shape, stride or workspace
        # scratch fields are changed.
        before = bytes(self.original_consumer.buffers[0])
        after = bytes(self.cached_consumer.buffers[0])
        assert before[:byte] == after[:byte] and before[byte + 8:] == after[byte + 8:]
        self.refresh_graph, self.refresh_exec = C.c_void_p(), C.c_void_p()
        api.check(api.lib.cuGraphCreate(C.byref(self.refresh_graph), 0), 'create refresh graph')
        node = C.c_void_p()
        api.check(api.lib.cuGraphAddKernelNode(C.byref(node), self.refresh_graph, None, 0,
                                             C.byref(self.transform.params)), 'clone native weight transform')
        api.check(api.lib.cuGraphInstantiateWithFlags(C.byref(self.refresh_exec), self.refresh_graph, 0),
                  'instantiate refresh graph')
        self.refresh_if_needed()
        self.proof = dict(transform=transform['name'], consumer=consumer['name'],
                          consumer_argument=0, consumer_pointer_byte=byte,
                          cached_bytes=self.cache.numel() * self.cache.element_size(),
                          preserved_intervening_memset_nodes=passthrough,
                          only_consumer_filter_pointer_changed=True, original_native_transform_cloned=True)

    def refresh_if_needed(self):
        assert not self.closed and tuple(self.weight.shape) == self.initial_shape
        assert self.weight.data_ptr() == self.initial_ptr and self.weight.is_contiguous()
        version = self.weight._version
        if version != self.version:
            self.api.check(self.api.lib.cuGraphLaunch(self.refresh_exec,
                C.c_void_p(torch.cuda.current_stream(self.weight.device).cuda_stream)), 'refresh native weight cache')
            self.cache.record_stream(torch.cuda.current_stream(self.weight.device))
            self.version = version
            self.refreshes += 1

    def verify_cache(self):
        co, ci = self.weight.shape[:2]
        expected = self.weight.reshape(co, ci, -1).permute(0, 2, 1).contiguous()
        actual = self.cache.reshape_as(expected)
        assert torch.isfinite(actual).all()
        bits = expected.view(torch.int32)
        rounded = ((bits + 0xfff + ((bits >> 13) & 1)) & -8192).view(torch.float32)
        ties_away = ((bits + 0x1000) & -8192).view(torch.float32)
        plain = torch.equal(actual.view(torch.int32), bits)
        tf32_rne = torch.equal(actual.view(torch.int32), rounded.view(torch.int32))
        tf32_rna = torch.equal(actual.view(torch.int32), ties_away.view(torch.int32))
        prebias = torch.equal(actual.view(torch.int32), bits + 0x1000)
        print('NATIVE_WEIGHT_CACHE_LAYOUT ' + str(dict(plain=plain, tf32_rne=tf32_rne, tf32_rna=tf32_rna,
              native_integer_prebias_exact=prebias,
              max_plain_error=(actual-expected).abs().max().item(),
              max_tf32_rne_error=(actual-rounded).abs().max().item(),
              max_tf32_rna_error=(actual-ties_away).abs().max().item())), flush=True)
        # Measured native preprocessing adds 0x1000 to FP32 integer bits without
        # masking low bits. Check it independently of the original-workspace
        # exact oracle; do not substitute a guessed TF32 cast convention.
        assert prebias
        self.verify_canaries()

    def verify_canaries(self):
        assert torch.all(self.backing[:self.guard_elements] == 90909.)
        assert torch.all(self.backing[-self.guard_elements:] == 90909.)

    def verify_original_transform(self):
        assert not self.enabled
        # Re-run the unchanged native graph, then copy its original filter
        # workspace before another convolution can reuse that workspace.
        self.graph.replay()
        original = torch.empty_like(self.cache)
        self.api.check(self.api.lib.cuMemcpyDtoDAsync_v2(original.data_ptr(), self.old_output,
            original.numel() * original.element_size(),
            C.c_void_p(torch.cuda.current_stream(self.weight.device).cuda_stream)),
            'copy original native transformed filter')
        assert torch.equal(original.view(torch.int32), self.cache.view(torch.int32))

    def set_cached(self, enabled):
        assert not self.closed and not torch.cuda.is_current_stream_capturing()
        if enabled:
            self.refresh_if_needed()
        params = self.cached_consumer if enabled else self.original_consumer
        self.api.check(self.api.lib.cuGraphExecKernelNodeSetParams(self.executable, self.consumer_node,
                        C.byref(params.params)), 'set native consumer filter pointer')
        try:
            self.api.check(self.api.lib.cuGraphNodeSetEnabled(self.executable, self.transform_node,
                            0 if enabled else 1), 'toggle original weight transform')
        except Exception:
            self.api.lib.cuGraphExecKernelNodeSetParams(self.executable, self.consumer_node,
                                                       C.byref(self.original_consumer.params))
            self.api.lib.cuGraphNodeSetEnabled(self.executable, self.transform_node, 1)
            self.enabled = False
            raise
        value = C.c_uint()
        self.api.check(self.api.lib.cuGraphNodeGetEnabled(self.executable, self.transform_node, C.byref(value)),
                       'verify weight transform enable state')
        assert bool(value.value) == (not enabled)
        self.enabled = enabled

    def replay(self):
        assert not self.closed
        if self.enabled:
            self.refresh_if_needed()
            self.cache.record_stream(torch.cuda.current_stream(self.weight.device))
        self.graph.replay()

    def poison_old_workspace(self):
        assert self.enabled
        self.api.check(self.api.lib.cuMemsetD32Async(self.old_output, 0x7fc00000, self.weight.numel(),
            C.c_void_p(torch.cuda.current_stream(self.weight.device).cuda_stream)), 'poison old transformed filter workspace')

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.weight.device)
        self.set_cached(False)
        self.api.check(self.api.lib.cuGraphExecDestroy(self.refresh_exec), 'destroy refresh exec')
        self.api.check(self.api.lib.cuGraphDestroy(self.refresh_graph), 'destroy refresh graph')
        self.closed = True
