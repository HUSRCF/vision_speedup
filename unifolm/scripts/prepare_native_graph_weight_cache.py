"""Host graph plumbing; preserve all previous Triton bodies and line positions."""
import ast


def one(text, old, new):
    assert text.count(old)==1, old
    return text.replace(old,new)


def transform(original):
    text=one(original, '            entry["graph"].replay()',
        '''            if entry.get("native_weight_cache") is not None:
                entry["native_weight_cache"].replay()
            else:
                entry["graph"].replay()''')
    text=one(text, '        graph = torch.cuda.CUDAGraph()',
        '''        graph = torch.cuda.CUDAGraph()
        from .native_graph_weight_cache import requested
        cache_weights = requested()
        weight_cache = None
        if cache_weights:
            graph.enable_debug_mode()''')
    text=one(text, '            graph.replay()',
        '''            if cache_weights:
                from .native_graph_weight_cache import NativeGraphWeights
                weight_cache = NativeGraphWeights(graph, self.original.__self__)
                weight_cache.replay()
            else:
                graph.replay()''')
    text=one(text, '            for actual, reference in zip(output_leaves, expected_leaves):\n                torch.testing.assert_close(actual, reference, rtol=0, atol=0)',
        '''            for actual, reference in zip(output_leaves, expected_leaves):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                if cache_weights and isinstance(actual, torch.Tensor) and actual.dtype == torch.float32:
                    assert torch.equal(actual.contiguous().view(torch.int32), reference.contiguous().view(torch.int32))''')
    text=one(text, '        return {"graph": graph, "leaves": static_leaves, "outputs": outputs,',
        '        return {"graph": graph, "native_weight_cache": weight_cache, "leaves": static_leaves, "outputs": outputs,')
    ast.parse(text)
    return text


def integration():
    return ''
