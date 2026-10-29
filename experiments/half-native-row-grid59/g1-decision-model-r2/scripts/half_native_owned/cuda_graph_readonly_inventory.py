"""Read native graph kernel argument layouts through CUDA's metadata APIs."""
import ctypes as C


class KernelParams(C.Structure):
    # CUDA_KERNEL_NODE_PARAMS_v1, paired with the unversioned legacy driver
    # symbol. The toolkit's C macro maps to _v2; ctypes does not expand it.
    _fields_ = [('func', C.c_void_p),
                ('grid_x', C.c_uint), ('grid_y', C.c_uint), ('grid_z', C.c_uint),
                ('block_x', C.c_uint), ('block_y', C.c_uint), ('block_z', C.c_uint),
                ('shared', C.c_uint), ('kernel_params', C.POINTER(C.c_void_p)),
                ('extra', C.POINTER(C.c_void_p))]


class Inventory:
    def __init__(self):
        self.lib = C.CDLL('libcuda.so.1')
        specs = {
            'cuGetErrorName': [C.c_int, C.POINTER(C.c_char_p)],
            'cuGraphGetNodes': [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
            'cuGraphGetEdges': [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
            'cuGraphNodeGetType': [C.c_void_p, C.POINTER(C.c_int)],
            'cuGraphChildGraphNodeGetGraph': [C.c_void_p, C.POINTER(C.c_void_p)],
            'cuGraphKernelNodeGetParams': [C.c_void_p, C.POINTER(KernelParams)],
            'cuFuncGetName': [C.POINTER(C.c_char_p), C.c_void_p],
            'cuFuncGetParamInfo': [C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)],
        }
        for name, types in specs.items():
            fn = getattr(self.lib, name)
            fn.argtypes = types
            fn.restype = C.c_int

    def check(self, status, name):
        if status:
            error = C.c_char_p()
            self.lib.cuGetErrorName(status, C.byref(error))
            raise RuntimeError(name + ': ' + str(error.value))

    def read(self, graph, tensors, depth=0):
        assert depth < 8
        known = {name: (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                 for name, t in tensors.items()}
        count = C.c_size_t()
        self.check(self.lib.cuGraphGetNodes(graph, None, C.byref(count)), 'node count')
        assert count.value < 10000
        array = (C.c_void_p * count.value)()
        self.check(self.lib.cuGraphGetNodes(graph, array, C.byref(count)), 'nodes')
        edge_count = C.c_size_t()
        self.check(self.lib.cuGraphGetEdges(graph, None, None, C.byref(edge_count)), 'edge count')
        sources = (C.c_void_p * edge_count.value)()
        destinations = (C.c_void_p * edge_count.value)()
        if edge_count.value:
            self.check(self.lib.cuGraphGetEdges(graph, sources, destinations, C.byref(edge_count)), 'edges')
        nodes = []
        for node in array:
            kind = C.c_int()
            self.check(self.lib.cuGraphNodeGetType(node, C.byref(kind)), 'node type')
            record = dict(node=hex(node), type=kind.value)
            if kind.value == 0:
                params = KernelParams()
                self.check(self.lib.cuGraphKernelNodeGetParams(node, C.byref(params)), 'kernel params')
                name = C.c_char_p()
                self.check(self.lib.cuFuncGetName(C.byref(name), params.func), 'function name')
                record.update(name=name.value.decode(), function=hex(params.func),
                              grid=[params.grid_x, params.grid_y, params.grid_z],
                              block=[params.block_x, params.block_y, params.block_z],
                              shared_bytes=params.shared, arguments=[], matched_tensors=[])
                if params.kernel_params:
                    for index in range(256):
                        offset, size = C.c_size_t(), C.c_size_t()
                        status = self.lib.cuFuncGetParamInfo(params.func, index, C.byref(offset), C.byref(size))
                        if status == 1:  # CUDA_ERROR_INVALID_VALUE: past the argument list.
                            break
                        self.check(status, 'argument metadata')
                        assert 0 < size.value <= 65536 and offset.value + size.value <= 65536
                        assert params.kernel_params[index]
                        raw = C.string_at(params.kernel_params[index], size.value)
                        words = []
                        for byte in range(0, len(raw) - 7, 8):
                            value = int.from_bytes(raw[byte:byte + 8], 'little')
                            matches = [key for key, (begin, end) in known.items() if begin <= value < end]
                            words.append(dict(byte=byte, value=hex(value), matched_tensors=matches))
                            record['matched_tensors'].extend(matches)
                        record['arguments'].append(dict(index=index, offset=offset.value,
                                                        size=size.value, bytes_hex=raw.hex(), words=words))
                    else:
                        raise RuntimeError('Argument list exceeds bounded inventory')
                else:
                    record['arguments_not_decoded'] = True
                record['matched_tensors'] = sorted(set(record['matched_tensors']))
            elif kind.value == 4:
                child = C.c_void_p()
                self.check(self.lib.cuGraphChildGraphNodeGetGraph(node, C.byref(child)), 'child graph')
                record['child'] = self.read(child.value, tensors, depth + 1)
            nodes.append(record)
        return dict(graph=hex(graph), nodes=nodes,
                    edges=[[hex(a), hex(b)] for a, b in zip(sources, destinations)])
