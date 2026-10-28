"""Numerical SGEMM-only candidate; requires full-video quality validation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import time
import torch
import triton
from .native_graph_weight_cache_probe import GraphAPI
from .native_graph_world_filter_cache_probe import FILTER,eligible_world_plan
from .native_winograd_gemm_fp16_kernel import transformed_gemm_fp16
from .native_sgemm_prepacked_swap import PrepackedSgemmSwap,NativeOnlyFilterSwap
from .native_kahan_filter_swap import CachedKahanFilterSwap
from .native_winograd_gemm_fp16_packed_kernel import transformed_gemm_fp16_packed


def requested():
    mode=os.environ.get('UNIFOLM_NATIVE_SGEMM_FP16','0')
    if mode not in {'0','1'}:raise ValueError('Invalid numerical native SGEMM mode')
    if mode=='1' and os.environ.get('UNIFOLM_NATIVE_WORLD_FILTER_CACHE','0')!='0':
        raise ValueError('Numerical SGEMM and world filter cache both update a native SGEMM node')
    return (mode=='1' and not torch.is_grad_enabled() and not torch.backends.cudnn.allow_tf32
            and not torch.backends.cuda.matmul.allow_tf32)


class NativeSgemmModel:
    def __init__(self,graph,model):
        assert requested() and torch.__version__=='2.3.1+cu121'
        assert os.environ.get('UNIFOLM_NATIVE_WORLD_FILTER_CACHE','0')=='0'
        so=Path(__file__).parent/'native-graph-handles/team07_cuda_graph_handles.so'
        assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
        spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so)
        extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
        api=GraphAPI();raw,_=extension.get_handles(graph);native=graph
        inventory=api.read(raw,{})
        swaps=[];templates={}
        configs={(320, 2560, 320): [32, 128, 64, 4, 1], (640, 640, 640): [64, 64, 64, 4, 1], (1280, 192, 1280): [64, 64, 64, 4, 1], (320, 2560, 960): [32, 128, 64, 4, 1], (640, 640, 1920): [64, 64, 64, 4, 1], (1280, 192, 2560): [64, 64, 64, 4, 1], (1280, 640, 1280): [32, 128, 64, 4, 1]}
        additional_configs={(640, 640, 320): {'config': [32, 64, 32, 4, 1], 'witness_cubin_sha256': '2daa957ff67201a6158b53f025ade1b002b5b2156db96e512f0ed51488fd312f', 'primitive_strict_conv_3e6_passed': False}, (1280, 192, 640): {'config': [32, 64, 32, 4, 1], 'witness_cubin_sha256': '8f16557ec155e0480eb4305ed8b3a7adaaffacb97986332af87f6827595838eb', 'primitive_strict_conv_3e6_passed': False}, (1280, 192, 1920): {'config': [32, 64, 32, 4, 1], 'witness_cubin_sha256': 'd255b2d7c2386c2812154e8e8b52a39e12faca771f1d0f4536692429a6fad2e5', 'primitive_strict_conv_3e6_passed': False}, (640, 640, 1280): {'config': [32, 64, 32, 4, 1], 'witness_cubin_sha256': 'db5bcb8c4b233404a3ffbc28c3ed80d7e670f249a085603eabee34a559c345a7', 'primitive_strict_conv_3e6_passed': False}, (640, 640, 960): {'config': [32, 64, 32, 4, 1], 'witness_cubin_sha256': '490ccb3992f302fa1e2bc4ad85765da525865ae7bdba9e9e6ebd2cc02191cebf', 'primitive_strict_conv_3e6_passed': False}, (640, 2560, 640): {'config': [32, 64, 32, 4, 1], 'witness_cubin_sha256': 'e69f99bebccb2402c1e2985cf226c2da73b4adf507e396dc3c1cca77d47459aa', 'primitive_strict_conv_3e6_passed': False}, (320, 2560, 640): {'config': [32, 64, 32, 4, 1], 'witness_cubin_sha256': '8131985e717553e86bf86993df382b1eddaf7416d28a2c5f2b6032ffc2fa575d', 'primitive_strict_conv_3e6_passed': False}}
        nodes={n['node']:n for n in inventory['nodes']};edges=inventory['edges']
        masters={w.data_ptr():(name,w) for name,w in model.named_parameters() if w.dtype==torch.float32 and w.is_cuda and w.is_contiguous() and w.ndim==4}
        chosen={};plans=[];started=time.perf_counter()
        for n in nodes.values():
            if n.get('name')!=FILTER or [a['size'] for a in n.get('arguments',[])]!=[40]:continue
            ptr=struct.unpack_from('<Q',bytes.fromhex(n['arguments'][0]['bytes_hex']),16)[0]
            if ptr not in masters:continue
            name,w=masters[ptr]
            outgoing=[b for a,b in edges if a==n['node']]
            view=dict(nodes=[dict(n,matched_tensors=['weight']),*[nodes[b] for b in outgoing]],edges=[[n['node'],b] for b in outgoing])
            plan=eligible_world_plan(view,w)
            if plan is None:continue
            _,consumer,_,_=plan;payload=bytes.fromhex(consumer['arguments'][0]['bytes_hex'])
            co,ci=w.shape[:2];stride=struct.unpack_from('<Q',payload,56)[0]
            assert stride%ci==0;m=stride//ci;geometry=(co,m,ci)
            if geometry not in configs:
                swap=(CachedKahanFilterSwap(native,w,plan,geometry,additional_configs[geometry],extension,api)
                    if geometry in additional_configs else NativeOnlyFilterSwap(native,w,plan,extension,api))
                swaps.append(swap);chosen[ptr]=w
                plans.append(dict(weight=name,geometry=geometry,config=None,**swap.proof))
                continue
            a,b,c=struct.unpack_from('<3Q',payload)
            assert b-a==36*m*ci*4 and c-b==36*co*ci*4
            assert struct.unpack_from('<3Q',payload,56)==(m*ci,ci*co,m*co)
            if geometry not in templates:
                bm,bn,bk,warps,stages=configs[geometry]
                dummy=torch.empty((1,),device=w.device)
                half_dummy=torch.empty((1,),device=w.device,dtype=torch.float16)
                kernel=transformed_gemm_fp16_packed.warmup(half_dummy,half_dummy,dummy,dummy,co,m,ci,bm,bn,bk,
                    num_warps=warps,num_stages=stages,enable_fp_fusion=False,
                    grid=(triton.cdiv(co,bm)*triton.cdiv(m,bn),36))
                kernel._init_handles()
                templates[geometry]=(None,half_dummy,dummy,kernel)
            kernel=templates[geometry][-1]
            swap=PrepackedSgemmSwap(native,w,plan,kernel,geometry,configs[geometry],extension,api)
            swaps.append(swap);chosen[ptr]=w
            plans.append(dict(weight=name,geometry=geometry,config=configs[geometry],**swap.proof))
        assert swaps,'No proven microbenchmark geometry selected'
        torch.cuda.synchronize();setup_ms=(time.perf_counter()-started)*1000
        self.graph,self.swaps,self.templates=graph,swaps,templates
        self.closed=False
        for swap in swaps:swap.set_enabled(True)
        self.report=dict(selected=len(swaps),plans=plans,setup_ms=setup_ms,
            prepacked_weights=True,direct_params=True,
            numerical_selected=sum(isinstance(s,PrepackedSgemmSwap) for s in swaps),
            native_only_selected=sum(isinstance(s,NativeOnlyFilterSwap) for s in swaps),
            kahan_selected=sum(isinstance(s,CachedKahanFilterSwap) for s in swaps),
            cached_native_filter_bytes=sum(s.cache.cache.numel()*4 for s in swaps),
            packed_filter_bytes=sum(s.wh.numel()*4 for s in swaps if isinstance(s,PrepackedSgemmSwap)),
            template_bytes=sum(row[1].numel()*2+row[2].numel()*4 for row in templates.values()),
            kahan_template_bytes=sum(s.dummy.numel()*4 for s in swaps if isinstance(s,CachedKahanFilterSwap)),
            original_native_transforms=True,numerical_SGEMM_only=True,
            first_setup_inside_timed_capture=True,full_video_PSNR_required=True)
        print('NATIVE_SGEMM_FP16 '+json.dumps(self.report),flush=True)
        from .world_head_filter_cache import attach
        attach(self,model,extension,api,inventory)

    def replay(self):
        assert not self.closed
        for swap in self.swaps:swap.refresh_if_needed()
        self.graph.replay()

    def close(self):
        if self.closed:return
        torch.cuda.synchronize()
        for swap in self.swaps:swap.close()
        for cg,*_ in self.templates.values():
            if cg is not None:cg.reset()
        self.closed=True
