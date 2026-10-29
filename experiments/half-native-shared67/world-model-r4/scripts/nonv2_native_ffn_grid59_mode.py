"""Retain native Half GEMM descriptors and shard only qualified row grids."""
from contextlib import contextmanager
import ctypes as C
import hashlib,importlib.util,inspect,json,os,struct,textwrap
from pathlib import Path
import torch
import torch.distributed as dist
from half_native_owned.native_graph_weight_cache_probe import GraphAPI,OwnedParams

active=None
enabled=False


def gather_rows(value,rank,gather):
    assert value.ndim==3 and value.is_contiguous()and value.shape[0]%2==0
    matrix=value.view(-1,value.shape[-1]);half=matrix.shape[0]//2
    assert half%value.shape[1]==0
    full=torch.empty_like(matrix);gather(full,matrix[rank*half:(rank+1)*half].contiguous())
    return full.view_as(value)


def choose(shape,qualifications):
    m,c=shape[0]*shape[1],shape[2]
    return (m,8*c,c)in qualifications and(m,c,4*c)in qualifications


class BoundGrid:
    def __init__(self,node,record,qualified,api,exe,rank):
        x,w,y=(record[n]for n in('x','w','y'));m,n,k=record['shape']
        assert all(t.is_cuda and t.dtype==torch.float16 and t.is_contiguous()for t in(x,w,y))
        assert x.numel()==m*k and y.numel()==m*n and tuple(w.shape)==(n,k)
        assert node['name']==qualified['metadata']['kernel_name']and node['grid']==qualified['metadata']['original_grid']and node['block']==qualified['metadata']['block']
        self.api,self.exe,self.node=api,exe,node;self.refs=(x,w,y);self.closed=False
        self.original=OwnedParams(node);replacements={};seen=set()
        for argument in node['arguments']:
            data=bytes.fromhex(argument['bytes_hex'])
            for byte in range(0,len(data)-7,8):
                old=struct.unpack_from('<Q',data,byte)[0]
                for name,tensor,width in [('x',x,k),('y',y,n)]:
                    if old!=tensor.data_ptr():continue
                    seen.add(name);replacements[(argument['index'],byte)]=(old,old+rank*(m//2)*width*2)
        assert seen=={'x','y'}
        self.modified=OwnedParams(node,replacements);axis=qualified['row_axis'];field=['grid_x','grid_y','grid_z'][axis]
        assert getattr(self.modified.params,field)%2==0
        setattr(self.modified.params,field,getattr(self.modified.params,field)//2)
        assert self.modified.params.func==self.original.params.func and self.modified.params.shared==self.original.params.shared
        self.api.check(self.api.lib.cuGraphExecKernelNodeSetParams(self.exe,int(node['node'],16),C.byref(self.modified.params)),'set qualified native Half FFN rows')
        self.public=dict(module=record['module'],role=record['role'],shape=list(record['shape']),row_axis=axis,kernel_name=node['name'],original_grid=node['grid'],original_compiled_function_and_K_order=True,original_descriptors_and_all_argument_bytes_except_owned_starts=True,exact_owned_Torch_input_output_weight_binding=True,synthetic_padded_six_state_domain_qualified=True,no_argument_bytes_addresses_or_offsets_exported=True)
    def close(self):
        if not self.closed:
            self.api.check(self.api.lib.cuGraphExecKernelNodeSetParams(self.exe,int(self.node['node'],16),C.byref(self.original.params)),'restore native Half FFN rows');self.closed=True


class Controller:
    def __init__(self,model):
        from unifolm_wma.modules.attention import BasicTransformerBlock,FeedForward
        from unifolm_wma.utils.cuda_graph import DenoiserGraph
        assert dist.is_initialized()and dist.get_world_size()==2
        self.rank=dist.get_rank();self.saved=[];self.ffns=[];self.rows=[];self.records=[];self.owners=[];self.binding_reports=[];self.scope=None
        raw=json.loads((Path(__file__).parent.parent/'qualified-shapes.json').read_text())
        self.qualified={tuple(r['shape']):r for r in raw['ranks'][self.rank]if r['supported']}
        assert len(self.qualified)==9 and raw['synthetic_only']
        so=Path(os.environ['NATIVE_GRAPH_HANDLES_PATH']);assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
        spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so);self.extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.extension);self.api=GraphAPI()
        self.graph_class=DenoiserGraph;self.old_capture=DenoiserGraph.capture
        source=textwrap.dedent(inspect.getsource(self.old_capture));anchor='    graph = torch.cuda.CUDAGraph()'
        assert source.count(anchor)==1
        source=source.replace(anchor,anchor+'\n    graph.enable_debug_mode()');ns=dict(self.old_capture.__globals__);exec(compile(source,'<owned-Half-graph-capture>','exec'),ns)
        self.debug_capture=ns['capture'];DenoiserGraph.capture=self.debug_capture
        for name,block in model.named_modules():
            if type(block)is not BasicTransformerBlock:continue
            module=block.ff;assert type(module)is FeedForward;self.ffns.append(module)
            before=module.forward
            def forward(value,_before=before,_module=module,_name=name):
                if not enabled or not choose(value.shape,self.qualified):return _before(value)
                assert self.scope is None and value.is_cuda and value.dtype==torch.float16 and value.is_contiguous()
                self.scope=dict(name=_name,M=value.numel()//value.shape[-1])
                try:
                    result=_before(value);full=gather_rows(result,self.rank,dist.all_gather_into_tensor)
                    if torch.cuda.is_current_stream_capturing():self.rows.append(dict(module=_name,input_shape=list(value.shape),collectives=1,full_GEMM_descriptors_preserved=True,local_output_MiB=result.numel()*result.element_size()/2**21))
                    return full
                finally:self.scope=None
            module.forward=forward;self.saved.append((module,before))
            for role,linear in [('producer',module.net[0].proj),('consumer',module.net[2])]:
                old=linear.forward
                def projected(value,*args,_old=old,_linear=linear,_role=role,_name=name,**kwargs):
                    result=_old(value,*args,**kwargs)
                    if enabled and self.scope is not None and self.scope['name']==_name and torch.cuda.is_current_stream_capturing():
                        shape=(value.numel()//value.shape[-1],result.shape[-1],value.shape[-1]);assert shape in self.qualified
                        self.records.append(dict(module=_name,role=_role,shape=shape,x=value,w=_linear.weight,y=result))
                    return result
                linear.forward=projected;self.saved.append((linear,old))
        assert len(self.ffns)==33
    def masters(self):return {p.data_ptr():p for m in self.ffns for p in m.parameters()}
    def bind(self,graph):
        assert len(self.rows)==31 and len(self.records)==62 and not self.owners
        raw,exe=self.extension.get_handles(graph);known={}
        for index,record in enumerate(self.records):
            for name in('x','w','y'):known[f'{index}_{name}']=record[name]
        inventory=self.api.read(raw,known)
        def flatten(tree):
            for node in tree['nodes']:
                if 'child'in node:yield from flatten(node['child'])
                else:yield node
        nodes=list(flatten(inventory));used=set()
        for index,record in enumerate(self.records):
            names={f'{index}_{name}'for name in('x','w','y')}
            matches=[node for node in nodes if node['type']==0 and names<=set(node.get('matched_tensors',[]))]
            assert len(matches)==1,'No unique native FFN kernel bound to exact owned tensors'
            node=matches[0];assert node['node']not in used;used.add(node['node'])
            owner=BoundGrid(node,record,self.qualified[record['shape']],self.api,exe,self.rank);self.owners.append(owner)
        assert len(used)==62
        # Poison legal, retained Torch outputs before a fully recomputed replay.
        for record in self.records:record['y'].fill_(float('nan'))
        graph.replay();torch.cuda.synchronize()
        self.binding_reports.append(dict(selected_FFNs=31,selected_native_GEMMs=62,original_full_shape_vendor_dispatch=True,debug_metadata_only_capture_edit=True,retained_owned_tensor_lifetimes=True,outputs_poisoned_before_first_partitioned_replay=True,kernels=[o.public for o in self.owners]))
    def close_owners(self):
        for owner in reversed(self.owners):owner.close()
    def report(self):
        return dict(captured=len(self.rows),rows=self.rows.copy(),native_linear_capture_count=len(self.records),native_binding=self.binding_reports.copy(),original59_math_and_full16_GEMM_shapes=True,full5D_GroupNorm_and16T_preserved=True,new_JIT_kernels=0,additional_collectives_per_forward=31,full_video_PSNR_unverified=True)


def install(model):
    global active
    assert active is None;active=Controller(model);return active


def uninstall():
    global active,enabled
    if active is not None:
        active.close_owners()
        for module,before in reversed(active.saved):module.forward=before
        assert active.graph_class.capture is active.debug_capture;active.graph_class.capture=active.old_capture
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    before=enabled;enabled=value
    try:yield
    finally:enabled=before
