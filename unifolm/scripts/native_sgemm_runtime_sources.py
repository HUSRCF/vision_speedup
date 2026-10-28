"""Prepare a separate numerical runtime only after real-model gates pass."""
import ast
import textwrap
from pathlib import Path


def runtime(configs,direct=False):
    if direct:
        from native_sgemm_direct_profile_sources import profile as direct_profile
        profile=direct_profile()
    else:profile=Path(__file__).with_name('profile_native_sgemm_pair.py').read_text()
    start=profile.index('                nodes={')
    end=profile.index('                def enabled(on):',start)
    setup=textwrap.dedent(profile[start:end])
    # These exact observed setup operations are shared with the diagnostic;
    # only the module-qualified helper imports differ in production.
    header='''"""Numerical SGEMM-only candidate; requires full-video quality validation."""
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
from .native_winograd_sgemm_swap import NativeSgemmSwap


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
'''
    source=header+'        configs='+repr({tuple(r['geometry']):r['config'] for r in configs})+'\n'
    source+=textwrap.indent(setup,'        ')
    source+='''        self.graph,self.swaps,self.templates=graph,swaps,templates
        self.closed=False
        for swap in swaps:swap.set_enabled(True)
        self.report=dict(selected=len(swaps),plans=plans,setup_ms=setup_ms,
            template_bytes=sum(t.numel()*4 for row in templates.values() for t in row[1:4]),
            original_native_transforms=True,numerical_SGEMM_only=True,
            first_setup_inside_timed_capture=True,full_video_PSNR_required=True)
        print('NATIVE_SGEMM_FP16 '+json.dumps(self.report),flush=True)

    def replay(self):
        assert not self.closed
        self.graph.replay()

    def close(self):
        if self.closed:return
        torch.cuda.synchronize()
        for swap in self.swaps:swap.close()
        for cg,*_ in self.templates.values():cg.reset()
        self.closed=True
'''
    if direct:
        source=source.replace('from .native_winograd_sgemm_swap import NativeSgemmSwap',
            'from .native_sgemm_direct_params import DirectSgemmSwap')
        source=source.replace('template_bytes=sum(t.numel()*4 for row in templates.values() for t in row[1:4]),',
            'direct_params=True,template_bytes=sum(row[1].numel()*4 for row in templates.values()),')
        source=source.replace('        for cg,*_ in self.templates.values():cg.reset()',
            '        for cg,*_ in self.templates.values():\n            if cg is not None:cg.reset()')
    ast.parse(source);return source


def transform(text):
    def one(old,new):
        nonlocal text
        assert text.count(old)==1,old
        text=text.replace(old,new)
    one('            if entry.get("native_weight_cache") is not None:',
        '            if entry.get("numerical_sgemm") is not None:\n                entry["numerical_sgemm"].replay()\n            elif entry.get("native_weight_cache") is not None:')
    one('        cache_weights = requested()\n        weight_cache = None\n        if cache_weights:',
        '        cache_weights = requested()\n        from .native_graph_sgemm import requested as numerical_requested\n        numerical_mode = numerical_requested()\n        numerical_sgemm = None\n        weight_cache = None\n        if cache_weights or numerical_mode:')
    anchor='        print(f"CUDA_GRAPH_CAPTURE index={len(self.entries)} "'
    one(anchor,'''        if numerical_mode:
            assert weight_cache is None
            from .native_graph_sgemm import NativeSgemmModel
            numerical_sgemm = NativeSgemmModel(graph, self.original.__self__)
            numerical_sgemm.replay()
            errors = []
            for actual, reference in zip(output_leaves, expected_leaves):
                assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
                delta = actual.double() - reference.double()
                relative = float(delta.norm() / reference.double().norm().clamp_min(1e-30))
                maximum = float(delta.abs().max())
                assert relative <= .001 and maximum <= .001 * max(1., float(reference.abs().max()))
                errors.append(dict(relative_l2=relative,max_abs=maximum))
            import json
            print('NATIVE_SGEMM_FIRST_CAPTURE '+json.dumps(dict(outputs=errors,
                native_capture_bitwise=True,numerical_replay=True,full_video_PSNR_required=True)),flush=True)
'''+anchor)
    one('              "eager_match=bitwise", flush=True)',
        '              + ("native_capture_match=bitwise replay_math=numerical" if numerical_mode else "eager_match=bitwise"), flush=True)')
    one('        return {"graph": graph, "native_weight_cache": weight_cache,',
        '        return {"graph": graph, "native_weight_cache": weight_cache, "numerical_sgemm": numerical_sgemm,')
    ast.parse(text);return text
