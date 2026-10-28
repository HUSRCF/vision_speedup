"""Production host launch policy; all existing Triton function bodies stay fixed."""
import ast

FLAG='UNIFOLM_TEMPORAL_NATIVE_W1'
SOURCE='src/unifolm_wma/utils/triton_temporal_attention_fp32.py'
MANIFEST='temporal-native-w1-production-sources.json'
SHAPES=((2560,16,512),(2560,16,320),(640,16,640),(160,16,1280),(40,16,1280))
DEPENDENCIES={'UNIFOLM_FUSED_TEMPORAL_ATTENTION_FP32':'1','UNIFOLM_NATIVE_FP32_TEMPORAL_ATTENTION':'0','UNIFOLM_CUDA_GRAPH':'1'}

def source(old):
    assert FLAG not in old and old.count('return_intermediates, num_warps=4, enable_fp_fusion=False)')==1
    helper='''
def _native_temporal_warps(query, heads, scale):
    flag = os.environ.get(FLAG, '0')
    if flag not in ('0', '1'):
        raise ValueError(FLAG + ' must be 0 or 1')
    if flag == '0' or os.environ.get('UNIFOLM_CASE') != 'unitree_z1_dual_arm_stackbox_v2':
        return 4
    if any(os.environ.get(k) != v for k, v in DEPENDENCIES.items()):
        raise ValueError(FLAG + ' requires its qualified inference dependencies')
    return (1 if tuple(query.shape) in QUALIFIED_SHAPES and
            query.shape[-1] == heads * 64 and scale == .125 and
            not torch.is_autocast_enabled() else 4)

'''.replace('FLAG',repr(FLAG)).replace('DEPENDENCIES',repr(DEPENDENCIES)).replace('QUALIFIED_SHAPES',repr(SHAPES))
    result=old.replace('import torch;', 'import os\nimport torch;',1)
    anchor='def fused_temporal_attention_fp32('
    assert result.count(anchor)==1
    result=result.replace(anchor,helper+anchor)
    result=result.replace('return_intermediates, num_warps=4, enable_fp_fusion=False)',
        'return_intermediates, num_warps=_native_temporal_warps(query, heads, scale), enable_fp_fusion=False)')
    ast.parse(result)
    return result

def provenance_source(old):
    marker='} > "${run_dir}/provenance.txt"'
    assert FLAG not in old and old.count(marker)==1
    return old.replace(marker,f'    echo "{FLAG}=${{{FLAG}:-0}}"\n'+marker)

def cpu_checker():
    return '''"""Exercise the actual production host function without a CUDA context."""
import os,json
from pathlib import Path
from unittest.mock import patch
import torch
from unifolm_wma.utils import triton_temporal_attention_fp32 as r
FLAG='UNIFOLM_TEMPORAL_NATIVE_W1'
SHAPES=QUALIFIED_SHAPES
DEPENDENCIES=QUALIFIED_DEPENDENCIES
class Tensor:
    is_cuda=True;dtype=torch.float32;device=torch.device('cuda:0');ndim=3
    def __init__(self,shape):self.shape=shape
    def is_contiguous(self):return True
class Kernel:
    def __getitem__(self,grid):
        def run(*args,**kwargs):calls.append((grid,args,kwargs))
        return run
calls=[]
env={FLAG:'1','UNIFOLM_CASE':'unitree_z1_dual_arm_stackbox_v2',**DEPENDENCIES}
with patch.dict(os.environ,env,clear=True),torch.no_grad():
    with patch.object(r,'_attention_fp32',Kernel()),patch.object(r,'empty_attention_output',lambda kind,q:q):
        for shape in SHAPES:
            q=Tensor(shape);heads=shape[2]//64
            assert r.fused_temporal_attention_fp32(q,q,q,heads,.125) is q
            grid,args,kwargs=calls[-1]
            assert grid==(shape[0]*heads,) and kwargs=={'num_warps':1,'enable_fp_fusion':False}
            assert all(v is q for v in args[:6]) and args[6:]==(heads,.125,False)
        q=Tensor(SHAPES[0]);heads=q.shape[-1]//64
        for extra in ({FLAG:'0'},{'UNIFOLM_CASE':'other'}):
            with patch.dict(os.environ,extra):
                r.fused_temporal_attention_fp32(q,q,q,heads,.125);assert calls[-1][2]['num_warps']==4
        for shape,scale in (((1,16,64),.125),(SHAPES[0],.25)):
            t=Tensor(shape);r.fused_temporal_attention_fp32(t,t,t,shape[-1]//64,scale);assert calls[-1][2]['num_warps']==4
        for extra in ({FLAG:'bad'},*({k:'bad'} for k in DEPENDENCIES)):
            before=len(calls)
            with patch.dict(os.environ,extra):
                try:r.fused_temporal_attention_fp32(q,q,q,heads,.125)
                except ValueError:pass
                else:raise AssertionError(extra)
            assert len(calls)==before
assert not torch.cuda.is_initialized()
report=dict(passed=True,CUDA_context_initialized=False,actual_host_function_tested=True,
            all5_grid_input_output_storage_and_scale_preserved=True,only_warps_changed=True,
            default_disabled=True,V2_only=True,invalid_flags_and_dependencies_rejected=True,
            unknown_shape_or_scale_original4=True)
Path('temporal-native-w1-production-cpu.json').write_text(json.dumps(report,indent=2)+'\\n')
print(json.dumps(report))
'''.replace('QUALIFIED_SHAPES',repr(SHAPES)).replace('QUALIFIED_DEPENDENCIES',repr(DEPENDENCIES))
