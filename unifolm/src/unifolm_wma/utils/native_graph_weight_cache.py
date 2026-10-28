"""Cache native cuDNN filter preprocessing without changing convolution math."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import torch
from .native_graph_weight_cache_probe import GraphAPI, NativeWeightCache, TRANSPOSE, eligible_plan


def requested():
    mode = os.environ.get('UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE', '0')
    if mode not in {'0', '1'}:
        raise ValueError('Invalid native graph weight cache mode')
    return (mode == '1' and not torch.is_grad_enabled()
            and torch.backends.cudnn.allow_tf32
            and not torch.backends.cuda.matmul.allow_tf32)


class NativeGraphWeights:
    def __init__(self, graph, model):
        assert requested() and torch.__version__ == '2.3.1+cu121'
        so = Path(__file__).parent / 'native-graph-handles/team07_cuda_graph_handles.so'
        assert hashlib.sha256(so.read_bytes()).hexdigest() == 'ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
        spec = importlib.util.spec_from_file_location('team07_cuda_graph_handles', so)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        api = GraphAPI()
        raw, _ = extension.get_handles(graph)
        inventory = api.read(raw, {})
        nodes = {n['node']: n for n in inventory['nodes']}
        successors = {}
        for a,b in inventory['edges']:
            successors.setdefault(a, []).append(b)
        masters = {p.data_ptr():(name,p) for name,p in model.named_parameters()
                   if p.dtype == torch.float32 and p.is_cuda and p.is_contiguous()
                   and p.ndim in [4,5]}
        self.graph, self.caches = graph, []
        self.stream = torch.cuda.current_stream()
        self.closed = False
        readers = 0
        try:
            for node in nodes.values():
                if node.get('name') != TRANSPOSE or len(node.get('arguments',[])) != 3:
                    continue
                ptr = int.from_bytes(bytes.fromhex(node['arguments'][1]['bytes_hex']), 'little')
                if ptr not in masters:
                    continue
                readers += 1
                name, weight = masters[ptr]
                outgoing = successors.get(node['node'], [])
                view_nodes = [dict(node,matched_tensors=['weight']), *[nodes[n] for n in outgoing]]
                edges = [[node['node'],n] for n in outgoing]
                frontier=outgoing
                for _ in range(2):
                    if len(frontier)!=1 or nodes[frontier[0]]['type']!=2:
                        break
                    after = successors.get(frontier[0],[])
                    view_nodes.extend(nodes[n] for n in after)
                    edges.extend([frontier[0],n] for n in after)
                    frontier=after
                plan = eligible_plan(dict(nodes=view_nodes,edges=edges), weight)
                if plan is None:
                    continue
                cache = NativeWeightCache(graph,weight,extension,api,validated_plan=plan)
                self.caches.append((name,cache))
            for _,cache in self.caches:
                cache.set_cached(True)
        except Exception:
            self.close()
            raise
        self.report = dict(selected=len(self.caches), native_weight_readers=readers,
                           cached_bytes=sum(c.cache.numel()*4 for _,c in self.caches),
                           graph_nodes=len(nodes), native_convolution_unchanged=True,
                           first_refresh_inside_timed_capture=True)
        print('NATIVE_GRAPH_WEIGHT_CACHE '+json.dumps(self.report), flush=True)
        from .native_conv1d_weight_cache import attach
        attach(self,model,extension,api,inventory)
        from .native_remaining_filter_cache import attach as attach_remaining
        attach_remaining(self,model,extension,api,inventory)

    def replay(self):
        assert not self.closed
        current = torch.cuda.current_stream()
        if current != self.stream:
            for _,cache in self.caches:
                cache.cache.record_stream(current)
            self.stream = current
        for _,cache in self.caches:
            cache.refresh_if_needed()
        self.graph.replay()

    def close(self):
        if self.closed:
            return
        for _,cache in self.caches:
            cache.close()
        self.closed = True
