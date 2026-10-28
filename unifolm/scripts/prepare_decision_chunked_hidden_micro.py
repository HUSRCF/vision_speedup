"""Derive exact C640 Half intermediates and chunk512 residual epilogue."""
import ast
from pathlib import Path
import sys

root=Path(sys.argv[1]);utils=root/'src/unifolm_wma/utils'


def replace(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new)


def extract(filename,name):
    source=(utils/filename).read_text();lines=source.splitlines(keepends=True)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    start=node.decorator_list[0].lineno if node.decorator_list else node.lineno
    return node,''.join(lines[start-1:node.end_lineno])


node,producer=extract('triton_primary_geglu.py','_primary_geglu')
producer=replace(producer,'def _primary_geglu(', 'def _primary_geglu_hidden_parts(')
producer=replace(producer,'Bias,Y,M:', 'Bias,YH,YL,M:')
producer=replace(producer,'''    tl.store(Y+rows[:,None]*C+cols[None,:],value*gelu,
             (rows[:,None]<M)&(cols[None,:]<C))''', '''    hidden=_rounding_boundary(value*gelu)
    high,low=_half_split(hidden,1.)
    valid=(rows[:,None]<M)&(cols[None,:]<C)
    tl.store(YH+rows[:,None]*C+cols[None,:],high,valid)
    tl.store(YL+rows[:,None]*C+cols[None,:],low,valid)''')
new=ast.parse(producer).body[0]
assert [ast.dump(n,include_attributes=False) for n in node.body[:-1]]==[ast.dump(n,include_attributes=False) for n in new.body[:len(node.body)-1]]
node,reduction=extract('cublas_chunked_compensated_linear.py','_reduce')
reduction=replace(reduction,'def _reduce(', 'def _reduce_residual(')
reduction=replace(reduction,'Correction,Bias,Output,', 'Correction,Bias,Residual,Output,')
reduction=replace(reduction,'    if GROUPS==1:', '''    acc=_rounding_boundary(acc)+tl.load(Residual+i,valid,other=0.)
    if GROUPS==1:''')
new=ast.parse(reduction).body[0]
assert [ast.dump(n,include_attributes=False) for n in node.body[:-2]]==[ast.dump(n,include_attributes=False) for n in new.body[:len(node.body)-2]]
_,consumer=extract('cublas_chunked_compensated_linear.py','chunked_compensated_linear')
consumer=replace(consumer,'def chunked_compensated_linear(value,parts,bias=None,*,chunk_k=256,',
                 'def chunked_hidden_residual(inputs,parts,residual,bias=None,*,chunk_k=512,')
consumer=replace(consumer,'    if (torch.is_grad_enabled()', '''    value=inputs[0]
    if len(inputs)!=2 or any(p.dtype!=torch.float16 or p.shape!=value.shape or p.device!=value.device
            or p.ndim!=2 or not p.is_contiguous() for p in inputs):
        raise ValueError('Expected fresh Half input parts')
    if residual.dtype!=torch.float32 or residual.device!=value.device or not residual.is_contiguous():
        raise ValueError('Expected original contiguous FP32 residual')
    if (torch.is_grad_enabled()''')
consumer=replace(consumer,'value.dtype!=torch.float32', 'value.dtype!=torch.float16')
consumer=replace(consumer,'    rows = value.numel()//inner', '''    rows = value.numel()//inner
    if residual.shape!=(rows,columns) or output_groups!=1:
        raise ValueError('Expected matching FP32 residual and one output group')''')
consumer=replace(consumer,'        xh,xl = pack_half_weight(value.view(rows,inner),weight_scale=1.)', '        xh,xl = inputs')
consumer=replace(consumer,'        _reduce[(', '        _reduce_residual[(')
consumer=replace(consumer,'parts[0] if bias is None else bias,output,rows,columns,chunks,',
                 'parts[0] if bias is None else bias,residual,output,rows,columns,chunks,')
source='''import ctypes
import torch
import triton
import triton.language as tl
from .triton_primary_geglu import _short_product
from .triton_half_compensated_linear import _half_split
from .triton_compensated_linear import _rounding_boundary
from .geglu_output_allocation import empty as empty_geglu
from .cublas_compensated_linear import _READY,_stream_key
from .cublas_chunked_compensated_linear import _batched_library,_batched_gemm,_written_buffer

'''+producer+'\n\n'+reduction+'\n\n'+consumer+'''


def primary_geglu_hidden_parts(inputs,parts,bias=None):
    high,low=inputs
    assert high.is_cuda and high.ndim==2 and high.dtype==low.dtype==torch.float16 and high.shape==low.shape
    rows,inner=high.shape;channels=parts[0].shape[0]//2
    assert inner==640 and channels==2560 and len(parts)==2
    assert all(p.shape==(channels*2,inner) and p.dtype==torch.float16 and p.device==high.device and p.is_contiguous() for p in parts)
    outputs=tuple(empty_geglu((rows,channels),device=high.device,dtype=torch.float16) for _ in range(2))
    _primary_geglu_hidden_parts[(triton.cdiv(rows,64)*triton.cdiv(channels,64),)](
        high,low,*parts,parts[0] if bias is None else bias,*outputs,
        rows,channels,inner,bias is not None,64,64,32,True,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return outputs
'''
ast.parse(source);(utils/'decision_chunked_hidden_pipeline.py').write_text(source)
print('DECISION_CHUNKED_HIDDEN_PREPARED GEGLU_prefix_AST=true chunk512_batched_GEMM_original=true ordered_reduce_prefix_AST=true no_model_install=true')
