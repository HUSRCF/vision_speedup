"""Host-only integration, preserving every pre-existing JIT body and line."""
import ast


def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def transform_pack(original):
    node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='pack_half_weight')
    lines=original.splitlines(keepends=True)
    body=''.join(lines[node.lineno-1:node.end_lineno])
    body=one(body,'    return parts','''    import os
    correction_mode=os.environ.get('UNIFOLM_DECISION_VENDOR_CORRECTION','0')
    if correction_mode not in {'0','1'}:
        raise ValueError('Invalid decision vendor correction mode')
    if correction_mode=='1' and weight_scale==256. and tuple(weight.shape)==(640,2560):
        from .vendor_correction_count import refresh_count
        refresh_count(parts)
    return parts''')
    result=''.join(lines[:node.lineno-1])+body+''.join(lines[node.end_lineno:])
    ast.parse(result);return result


def transform_pipeline(original):
    cut=original.index('def chunked_hidden_residual(')
    end=original.index('def primary_geglu_hidden_parts(',cut)
    prefix,body,suffix=original[:cut],original[cut:end],original[end:]
    body=one(body,'    with torch.cuda.device(value.device):','''    import os
    correction_mode=os.environ.get('UNIFOLM_DECISION_VENDOR_CORRECTION','0')
    if correction_mode not in {'0','1'}:
        raise ValueError('Invalid decision vendor correction mode')
    correction_count=None
    if (correction_mode=='1' and (rows,columns,inner,chunk_k,output_groups)==(10240,640,2560,512,1)
            and torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32):
        from .vendor_correction_count import get_count
        correction_count=get_count(parts)
    with torch.cuda.device(value.device):''')
    body=one(body,'        _batched_gemm(lib,handle,xh,parts[1],correction,chunk_k,accumulate=True)',
        '''        if correction_count is None:
            _batched_gemm(lib,handle,xh,parts[1],correction,chunk_k,accumulate=True)
        else:
            from .triton_predicated_chunked_correction import _third_product
            _third_product[(triton.cdiv(rows,64)*triton.cdiv(columns,64),chunks)](
                xh,parts[1],correction,correction_count,rows,columns,inner,chunk_k,64,64,
                num_warps=4,num_stages=3,enable_fp_fusion=False)''')
    result=prefix+body+suffix;ast.parse(result);return result


def integration():
    return '''    decision_vendor_correction = os.environ.get("UNIFOLM_DECISION_VENDOR_CORRECTION", "0")
    if decision_vendor_correction not in {"0", "1"}:
        raise ValueError("Invalid decision vendor correction mode")
    if decision_vendor_correction == "1":
        if sum(bool(getattr(m, "_decision_chunked_hidden_parts_installed", False))
                for m in model.model.diffusion_model.modules()) != 10:
            raise ValueError("Vendor correction requires the ten decision chunk512 blocks")
        print("DECISION_VENDOR_CORRECTION_MODEL mode=1 blocks=10 M=10240 N=640 K=2560 chunk=512 "
              "first_two_products=original_cuBLAS third_product=device_predicated_Triton "
              "nonzero_path=complete_chunk_MMA weights=first_timed_warm_refresh_either_stage_same_count_storage "
              "cold_stale_predicate_and_other_shapes_backend=original_cuBLAS "
              "FP32_reduce_bias_residual=previous world=previous", flush=True)
'''
