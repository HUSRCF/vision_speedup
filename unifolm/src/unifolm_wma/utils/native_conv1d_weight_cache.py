"""Cache native Conv1d filter transforms; preserve original graph ownership."""
import json,os,struct
import torch
from unifolm_wma.utils.native_graph_weight_cache_probe import TRANSPOSE,XMMA,XMMA_NONINDEXED,NativeWeightCache
CUTLASS_64='_ZN19cutlass_cudnn_infer6KernelI65cutlass_tensorop_s1688fprop_optimized_tf32_64x64_32x5_nhwc_align4EEvNT_6ParamsE'
def plan_for(nodes, outgoing, transform, weight):
    assert weight.ndim == 3 and weight.is_contiguous() and weight.dtype == torch.float32
    co, ci, spatial = weight.shape
    assert co % 32 == 0 and ci % 16 == 0 and spatial in (1, 3, 5)
    args = transform['arguments']
    assert [a['size'] for a in args] == [44, 8, 8]
    expected = (co, ci, ci, ci*spatial, spatial, spatial, ci, ci*spatial, ci, 0x3f800000, 0)
    assert struct.unpack('<11I', bytes.fromhex(args[0]['bytes_hex'])) == expected
    assert int.from_bytes(bytes.fromhex(args[1]['bytes_hex']), 'little') == weight.data_ptr()
    old = int.from_bytes(bytes.fromhex(args[2]['bytes_hex']), 'little')
    frontier = outgoing[transform['node']]
    path = []
    for _ in range(3):
        assert len(frontier) == 1
        consumer = nodes[frontier[0]]
        if consumer['type'] != 2:
            break
        path.append(consumer['node'])
        frontier = outgoing[consumer['node']]
    assert len(path) <= 2 and len(consumer.get('arguments', [])) == 1
    payload = bytes.fromhex(consumer['arguments'][0]['bytes_hex'])
    if consumer['name'].startswith((XMMA, XMMA_NONINDEXED)):
        assert len(payload) in (784, 1440)
        byte = 8
    else:
        assert consumer['name'] == CUTLASS_64 and len(payload) == 464
        byte = 240
    assert int.from_bytes(payload[byte:byte+8], 'little') == old
    return transform, consumer, byte, old, path

FLAG='UNIFOLM_NATIVE_CONV1D_WEIGHT_CACHE'
DEPENDENCIES={'UNIFOLM_CUDA_GRAPH': '1', 'UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE': '1', 'UNIFOLM_DENOISER_MODE': 'native', 'UNIFOLM_MATH_MODE': 'fp32'}


def configured(environment):
    raw=environment.get(FLAG,'0')
    if raw not in ('0','1'):raise ValueError('Invalid native Conv1d cache flag')
    if raw=='0' or environment.get('UNIFOLM_CASE')!='unitree_z1_dual_arm_stackbox_v2':return False
    for name,value in DEPENDENCIES.items():
        if environment.get(name)!=value:raise ValueError('Native Conv1d cache requires '+name+'='+value)
    return True


class Controller:
    def __init__(self,owner,model,extension,api,inventory):
        assert not torch.is_grad_enabled() and not torch.cuda.is_current_stream_capturing()
        assert torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        assert len(owner.caches)==138 and owner.report['native_weight_readers']==143
        assert not hasattr(owner,'_native_conv1d_weight_cache')
        self.owner,self.caches,self.closed=owner,[],False
        self.original_replay,self.original_close=owner.replay,owner.close
        self.old_caches=tuple(owner.caches)
        self.stream=torch.cuda.current_stream()
        nodes={n['node']:n for n in inventory['nodes']};outgoing={}
        for a,b in inventory['edges']:outgoing.setdefault(a,[]).append(b)
        parameters={id(p):name for name,p in model.named_parameters()}
        masters={m.weight.data_ptr():(parameters[id(m.weight)],m.weight) for m in model.modules()
                 if isinstance(m,torch.nn.Conv1d) and m.weight.dtype==torch.float32
                 and m.weight.is_cuda and m.weight.is_contiguous()}
        plans=[]
        for n in nodes.values():
            if n.get('name')!=TRANSPOSE:continue
            ptr=int.from_bytes(bytes.fromhex(n['arguments'][1]['bytes_hex']),'little')
            if ptr not in masters:continue
            name,weight=masters[ptr];plans.append((name,weight,plan_for(nodes,outgoing,n,weight)))
        assert len(plans)==43 and len({p.data_ptr() for _,p,_ in plans})==43
        assert not any(plan[1]['node']==hex(c.consumer_node) for _,_,plan in plans for _,c in owner.caches)
        try:
            for name,weight,plan in plans:
                self.caches.append((name,NativeWeightCache(owner.graph,weight,extension,api,validated_plan=plan)))
            self.set_enabled(True)
        except Exception:
            for _,c in self.caches:c.close()
            raise
        owner._native_conv1d_weight_cache=self
        owner.replay=self.replay;owner.close=self.close
        print('NATIVE_CONV1D_WEIGHT_CACHE '+json.dumps(dict(selected=43,
            cached_bytes=sum(c.cache.numel()*4 for _,c in self.caches),original_native_caches=138,
            original_native_transform_and_convolution=True,live_weight_version_refresh=True,
            no_activation_cache=True,all_setup_inside_process_wall=True)),flush=True)

    def set_enabled(self,enabled):
        assert not self.closed
        for _,c in self.caches:c.set_cached(enabled)
        self.enabled=enabled

    def replay(self):
        assert not self.closed and tuple(self.owner.caches)==self.old_caches
        if self.enabled:
            current=torch.cuda.current_stream()
            for _,c in self.caches:
                if current!=self.stream:c.cache.record_stream(current)
                c.refresh_if_needed()
            self.stream=current
        return self.original_replay()

    def close(self):
        if self.closed:return
        for _,c in self.caches:c.close()
        self.owner.replay=self.original_replay;self.owner.close=self.original_close
        self.closed=True
        self.original_close()


def attach(owner,model,extension,api,inventory):
    if not configured(os.environ):return None
    return Controller(owner,model,extension,api,inventory)
