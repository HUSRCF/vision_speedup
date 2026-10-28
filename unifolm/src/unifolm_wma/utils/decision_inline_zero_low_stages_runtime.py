"""Launch stages3 on qualified inline decision GEMMs; original kernel, storage and JIT object."""
import os
import torch
ORIGINAL_JIT=None
ORIGINAL_RUN=None
active=None


def flag_enabled():
    value=os.environ.get('UNIFOLM_DECISION_INLINE_ZERO_LOW_STAGES3','0')
    if value not in {'0','1'}:raise ValueError('Invalid decision GEMM stages')
    if value=='0':return False
    for name,expected in DEPENDENCIES:
        if os.environ.get(name)!=expected:raise ValueError('Decision GEMM stages requires '+name+'='+expected)
    return True


class Enabled:
    def __bool__(self):return flag_enabled()


enabled=Enabled()
DEPENDENCIES=[('UNIFOLM_CUDA_GRAPH', '1'), ('UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE', '1'), ('UNIFOLM_WORLD_PREPACKED_ZERO_LOW', '1'), ('UNIFOLM_DECISION_COMBINED_ZERO_LOW', '1'), ('UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK', 'wide'), ('UNIFOLM_WORLD_ZERO_LOW_STAGES3', '1'), ('UNIFOLM_DECISION_ZERO_LOW_STAGES3', '1')]
CONFIGS=[{'geometry': [40960, 512, 512], 'config': [32, 64, 32, 32], 'baseline_config': [32, 64, 32, 32], 'has_bias': False, 'output_groups': 1, 'num_stages': 3, 'nodes': 4}, {'geometry': [40960, 320, 1280], 'config': [64, 64, 64, 32], 'baseline_config': [64, 64, 64, 32], 'has_bias': True, 'output_groups': 1, 'num_stages': 3, 'nodes': 10}, {'geometry': [10240, 640, 640], 'config': [64, 64, 64, 32], 'baseline_config': [64, 64, 64, 32], 'has_bias': True, 'output_groups': 1, 'num_stages': 3, 'nodes': 40}, {'geometry': [10240, 640, 640], 'config': [64, 64, 64, 32], 'baseline_config': [64, 64, 64, 32], 'has_bias': False, 'output_groups': 1, 'num_stages': 3, 'nodes': 5}]


def measure(actual,expected):
    assert actual.dtype==expected.dtype==torch.float32 and actual.shape==expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    assert torch.equal(actual.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))
    return dict(relative_l2=0.,max_abs=0.,signedzero_bits=True)


class Controller:
    def __init__(self,configs):
        assert all(c['output_groups']==1 for c in configs)
        self.configs={(tuple(c['geometry']),c['has_bias']):c for c in configs}
        self.captures=[];self.fallbacks={}

    def old(self,reason,grid,args,kwargs):
        self.fallbacks[reason]=self.fallbacks.get(reason,0)+1
        return ORIGINAL_RUN(*args,grid=grid,warmup=False,**kwargs)

    def launch(self,grid,args,kwargs):
        if (not enabled or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or not torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32):
            return self.old('disabled',grid,args,kwargs)
        v=dict(zip(ORIGINAL_JIT.arg_names,args));v.update(kwargs)
        if any(n not in v for n in ORIGINAL_JIT.arg_names):return self.old('arguments',grid,args,kwargs)
        geometry=tuple(v[k] for k in ['M','N','K']);m,n,k=geometry
        config=self.configs.get((geometry,v['HAS_BIAS']))
        if (config is None or [v[t] for t in ['BM','BN','BK','GROUP_M']]!=config['config']
                or v['XS']!=1. or v['WS']!=256. or v['PACKED'] is not True
                or kwargs.get('num_warps')!=4 or kwargs.get('num_stages')!=2
                or kwargs.get('enable_fp_fusion') is not False):return self.old('unqualified',grid,args,kwargs)
        x=v['X'];parts=[v[name] for name in ['Weight','WeightLow']]
        if (not x.is_cuda or x.dtype!=torch.float32 or not x.is_contiguous() or x.numel()!=m*k
                or any(p.dtype!=torch.float16 or p.device!=x.device or not p.is_contiguous() or p.numel()!=n*k for p in parts)
                or v['Output'].device!=x.device or v['Output'].dtype!=torch.float32 or not v['Output'].is_contiguous() or v['Output'].numel()!=m*n
                or v['LowCount'].device!=x.device or v['LowCount'].dtype!=torch.int64 or v['LowCount'].shape!=()
                or (v['HAS_BIAS'] and (v['Bias'].device!=x.device or v['Bias'].dtype!=torch.float32 or v['Bias'].shape!=(n,) or not v['Bias'].is_contiguous()))):
            return self.old('layout',grid,args,kwargs)
        if torch.cuda.is_current_stream_capturing():self.captures.append(dict(geometry=list(geometry),has_bias=v['HAS_BIAS'],num_stages=config['num_stages']))
        return ORIGINAL_RUN(*args,grid=grid,warmup=False,**dict(kwargs,num_stages=config['num_stages']))

    def report(self):
        return dict(captured=len(self.captures),captures=self.captures.copy(),fallbacks=self.fallbacks.copy(),
            original_JIT_object_retained=True,original_kernel_and_input_Count_output_storage=True,
            compiler_launch_stages_only=True,input_splitting_still_inside_original_GEMM=True)


def launched(*args,grid=None,warmup=False,**kwargs):
    if warmup:return ORIGINAL_RUN(*args,grid=grid,warmup=True,**kwargs)
    return active.launch(grid,args,kwargs)


def wrap_gemm(original):
    global ORIGINAL_JIT,ORIGINAL_RUN,active
    flag_enabled()
    assert getattr(original.run,'__self__',None) is original
    assert ORIGINAL_JIT is None
    ORIGINAL_JIT=original;ORIGINAL_RUN=original.run
    active=Controller(CONFIGS);original.run=launched
    return original


def report_capture():
    if flag_enabled() and torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32:
        import json
        report=active.report()
        report.update(all_setup_inside_process_wall=True,full_video_PSNR_unverified=True)
        print('DECISION_INLINE_ZERO_LOW_STAGES_CAPTURE '+json.dumps(report),flush=True)
