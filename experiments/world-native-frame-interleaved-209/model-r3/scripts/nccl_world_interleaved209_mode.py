"""Whole-world diagnostic: shard the ResBlocks owning 21 native SGEMMs.

Keep original N16 descriptors and all production owner checks. Only a fresh,
captured, bounded-ABI-witnessed compensated GEMM replaces each eligible node.
The other two outer Upsample owners never participate in frame remapping.
"""
import ctypes as C
import hashlib,importlib.util,json,os
from contextlib import contextmanager
from pathlib import Path
import torch,triton
import torch.distributed as dist
from nccl_resblock_native_grid209_mode import Controller as Routing,patched_forward,measure
from prefix_kernel_witness209 import witness
from world_native_frame_interleaved209_kernel import (
    transformed_gemm_fp16_packed_frame_interleaved,
    transformed_gemm_fp16_kahan_frame_interleaved)
from unifolm_wma.utils.native_graph_weight_cache_probe import OwnedParams

enabled=False
active=None
SLOW={(1280,192,1280),(1280,192,2560),(1280,640,1280)}


def bind_owners(owner,names,geometry):
    assert len(owner.swaps)==owner.report['selected']==23
    assert (owner.report['numerical_selected'],owner.report['kahan_selected'])==(15,8)
    assert len(owner._world_head_filter_cache.caches)==45
    eligible=[];outside=[]
    for swap,plan in zip(owner.swaps,owner.report['plans']):
        local=names.get(id(swap.cache.weight))
        if local is None:
            assert plan['weight'].endswith(('output_blocks.5.3.conv.weight','output_blocks.8.3.conv.weight'))
            outside.append(plan['weight']);continue
        shape=geometry[local];co,cols,ci=plan['geometry']
        assert tuple(swap.cache.weight.shape)==(co,ci,3,3)
        assert (co,ci)==(shape['co'],shape['ci']) and shape['full_frames']==16
        assert cols==16*triton.cdiv(shape['height'],4)*triton.cdiv(shape['width'],4)
        eligible.append((local,swap,tuple(plan['geometry'])))
    assert len(eligible)==21 and len(outside)==2 and len({id(v[1]) for v in eligible})==21
    return eligible,outside


@contextmanager
def new_kernel_cache():
    # Do not install a custom cache manager: Triton2.3.1 retains that class
    # globally after the environment variable is removed. Change only the
    # directory during this new JIT's compile/launch, then restore it exactly.
    assert 'TRITON_CACHE_MANAGER' not in os.environ
    old=os.environ.get('TRITON_CACHE_DIR')
    root=Path(os.environ['PREFIX_CACHE_DIR']);assert root.is_dir()
    assert dist.is_initialized() and dist.get_world_size()==2
    # Triton2.3.1 CompiledKernel globs <stem>.* while FileCacheManager.put
    # writes <stem>.cubin.tmp.pid_*. The other rank can then read that
    # temporary binary as text. Isolate only the new kernels per rank.
    directory=root/('rank-'+str(dist.get_rank()))
    directory.mkdir(mode=0o700,exist_ok=True)
    os.environ['TRITON_CACHE_DIR']=str(directory)
    try:yield
    finally:
        if old is None:os.environ.pop('TRITON_CACHE_DIR',None)
        else:os.environ['TRITON_CACHE_DIR']=old
        assert os.environ.get('TRITON_CACHE_DIR')==old


class Controller:
    frame_rows=Routing.frame_rows
    embedding_rows=Routing.embedding_rows
    gather=Routing.gather

    def __init__(self,model,selection=None):
        from unifolm_wma.modules.networks.wma_model import ResBlock
        from unifolm_wma.utils.native_graph_sgemm import NativeSgemmModel
        assert dist.get_world_size()==2
        self.rank=dist.get_rank();self.current=None
        self.saved=[];self.hooks=[];self.captures=[];self.native_reports=[]
        self.geometry={};self.weight_names={};self.templates={};self.owners=[];self.replacements=[]
        root=Path(__file__).resolve().parent.parent
        if selection is None:
            scope=json.loads((root/'world-interleaved-owner-scope.json').read_text())
            names={r['weight'].removeprefix('model.diffusion_model.').rsplit('.',3)[0]
                for r in scope['rows'] if r['within_qualified_ResBlock']}
            assert len(names)==12
            selection=dict(names=sorted(names),expected_total=22)
        chosen=set(selection['names']);all_blocks={n for n,m in model.named_modules() if type(m) is ResBlock}
        assert len(all_blocks)==selection['expected_total'] and chosen<=all_blocks
        self.excluded=sorted(all_blocks-chosen);self.original_blocks=len(all_blocks)
        function=patched_forward(ResBlock._forward,self)
        for name,module in model.named_modules():
            if name not in chosen:continue
            assert type(module) is ResBlock
            old=module._forward
            def forward(value,emb,batch_size=None,_module=module,_old=old,_name=name):
                if not enabled or torch.backends.cudnn.allow_tf32:return _old(value,emb,batch_size=batch_size)
                assert not torch.is_grad_enabled() and not _module.training and not torch.is_autocast_enabled()
                assert not torch.backends.cuda.matmul.allow_tf32 and self.current is None
                assert value.ndim==4 and value.shape[0]==16 and value.dtype==torch.float32
                assert batch_size in (None,1)
                self.current=dict(module=_name,input_shape=list(value.shape),input_stride=list(value.stride()),
                    frame_start=self.rank*8,frames=8,logical_frames=16,gathered=False,embedding_rows=None)
                try:
                    output=function(_module,value,emb,batch_size=batch_size)
                    assert self.current['gathered'] and output.shape[0]==16
                    if torch.cuda.is_current_stream_capturing():self.captures.append(dict(self.current))
                    return output
                finally:self.current=None
            module._forward=forward;self.saved.append((module,old))
            for suffix,conv in module.named_modules():
                if not isinstance(conv,torch.nn.Conv2d) or conv.kernel_size!=(3,3):continue
                key=name+'.'+suffix+'.weight';self.weight_names[id(conv.weight)]=key
                def observe(conv,args,_name=key):
                    if self.current is None or not torch.cuda.is_current_stream_capturing():return
                    value=args[0];assert value.shape[0]==16 and value.shape[1]==conv.in_channels
                    record=dict(co=conv.out_channels,ci=conv.in_channels,height=value.shape[2],width=value.shape[3],
                        input_stride=list(value.stride()),full_frames=16)
                    assert _name not in self.geometry or self.geometry[_name]==record
                    self.geometry[_name]=record
                self.hooks.append(conv.register_forward_pre_hook(observe))
        assert len(self.saved)==len(chosen)
        self.owner_class=NativeSgemmModel;self.old_init=NativeSgemmModel.__init__
        def initialize(owner,graph,model):
            self.old_init(owner,graph,model)
            if not enabled:return
            self.owners.append(owner)
            assert self.current is None and len(self.geometry)==2*len(self.saved)
            eligible,outside=bind_owners(owner,self.weight_names,self.geometry)
            specs=json.loads((root/'matrix-geometries.json').read_text());specs={tuple(s['geometry']):s for s in specs}
            so=Path(os.environ['CANONICAL_ROOT'])/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so'
            assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
            spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so)
            extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
            rows=[]
            for local,swap,geometry in eligible:
                if geometry in SLOW:
                    rows.append(dict(weight=local,geometry=geometry,selected=False,reason='measured_slower_geometry'));continue
                config=specs[geometry]['config'];family=specs[geometry]['family']
                if geometry not in self.templates:self.templates[geometry]=self.template(geometry,config,family,extension,swap.api)
                kernel,proof=self.templates[geometry]
                old=swap.replacement;co,cols,k=geometry;bm,bn,bk,warps,stages=config
                assert len(old.buffers)==(4 if family=='packed' else 3) and all(len(b)==8 for b in old.buffers)
                replacement=OwnedParams(dict(function=hex(kernel.function),
                    grid=[triton.cdiv(co,bm)*triton.cdiv(cols//2,bn),36,1],block=[32*warps,1,1],shared_bytes=kernel.shared,
                    arguments=[dict(index=i,offset=i*8,size=8,bytes_hex=bytes(b).hex()) for i,b in enumerate(old.buffers)]))
                assert [bytes(b) for b in replacement.buffers]==[bytes(b) for b in old.buffers]
                self.replacements.append((swap,old,replacement));swap.replacement=replacement;swap.set_enabled(True)
                rows.append(dict(weight=local,geometry=geometry,selected=True,
                    original_owner_validated_pointer_bytes_unchanged=True,original_close_params_unchanged=True,
                    fresh_capture_ABI=proof))
            record=dict(original_SGEMM_owners=23,original_head_caches=45,eligible_ResBlock_owners=len(eligible),
                selected=sum(r['selected'] for r in rows),excluded_Upsample_owners=outside,
                original_CUBIN_and_cache_gates_completed_first=True,rows=rows)
            self.native_reports.append(record);print('WORLD_INTERLEAVED_OWNER '+json.dumps(record),flush=True)
        self.initialize=initialize;NativeSgemmModel.__init__=initialize

    def template(self,geometry,config,family,extension,api):
        co,cols,k=geometry;bm,bn,bk,warps,stages=config
        function=transformed_gemm_fp16_packed_frame_interleaved if family=='packed' else transformed_gemm_fp16_kahan_frame_interleaved
        # Full, properly typed allocations; never launch a full-size template
        # using one-element dummy buffers or guessed native-workspace tensors.
        if family=='packed':
            args=[torch.empty((36,co,k),device='cuda',dtype=torch.float16) for _ in range(2)]
        else:args=[torch.empty((36,co,k),device='cuda')]
        args.extend([torch.empty((36,k,cols),device='cuda'),torch.empty((36,co,cols),device='cuda')])
        grid=(triton.cdiv(co,bm)*triton.cdiv(cols//2,bn),36)
        kw=dict(num_warps=warps,num_stages=stages,enable_fp_fusion=False)
        with new_kernel_cache():
            kernel=function.warmup(*args,co,cols,k,bm,bn,bk,cols//2,grid=grid,**kw)
            kernel._init_handles();capture=torch.cuda.CUDAGraph();capture.enable_debug_mode()
            with torch.cuda.graph(capture):function[grid](*args,co,cols,k,bm,bn,bk,cols//2,**kw)
        try:proof=witness(capture,kernel,[a.data_ptr() for a in args],grid,warps,extension,api)
        finally:capture.reset()
        proof.update(capture_only_not_replayed=True,no_uninitialized_numeric_result_claim=True,
            proper_fullsize_known_Torch_allocations=True,default_old_kernel_cache_restored=True,
            new_kernel_cache_per_rank=True,new_kernel_cache_rank=dist.get_rank())
        return kernel,proof

    def report(self):
        assert self.current is None
        return dict(captured=len(self.captures),rows=self.captures.copy(),ResBlocks=len(self.saved),
            original_ResBlocks=self.original_blocks,excluded_blocks=self.excluded,
            excluded_reason='no_qualified_native_SGEMM_in_these_blocks',native_cache_coverage=self.native_reports.copy(),
            original_full_N16_descriptors=True,original_full16_temporal_and_heads=True,
            embedding_projection_full_shape=True,world_only=True,full_video_PSNR_unverified=True,two_GPUs=True)


def install(model,selection=None):
    global active
    assert active is None;active=Controller(model,selection);return active


def uninstall():
    global active,enabled
    if active is not None:
        assert active.current is None
        for owner in active.owners:
            if not owner.closed:owner.close()
        for swap,old,new in active.replacements:
            assert swap.closed and swap.replacement is new;swap.replacement=old
        for module,old in reversed(active.saved):module._forward=old
        for hook in active.hooks:hook.remove()
        assert active.owner_class.__init__ is active.initialize
        active.owner_class.__init__=active.old_init
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    prior=enabled;enabled=value
    try:yield
    finally:
        assert active is None or active.current is None
        enabled=prior
