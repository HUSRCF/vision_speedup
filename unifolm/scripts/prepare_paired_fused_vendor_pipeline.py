"""Fresh GEGLU planes feed the row-paired consumer without an activation copy."""
import ast


def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def paired_transform(original):
    cut=original.index('def chunked_hidden_residual(')
    split=original.index('def primary_geglu_hidden_parts(',cut)
    prefix,consumer,producer=original[:cut],original[cut:split],original[split:]
    consumer=one(consumer,'    with torch.cuda.device(value.device):','''    paired_mode=os.environ.get('UNIFOLM_DECISION_PAIRED_VENDOR','0')
    if paired_mode not in {'0','1'}:
        raise ValueError('Invalid decision paired vendor mode')
    if paired_mode=='1' and correction_count is not None:
        stack=getattr(inputs[0],'_paired_vendor_stack',None)
        if (isinstance(stack,torch.Tensor) and stack.shape==(2,rows,inner)
                and stack.dtype==torch.float16 and stack.device==value.device and stack.is_contiguous()
                and stack.data_ptr()==inputs[0].data_ptr()
                and inputs[1].data_ptr()==stack.data_ptr()+rows*inner*stack.element_size()):
            from .paired_vendor_hidden_pipeline import paired_hidden_residual
            with torch.cuda.device(value.device):
                return paired_hidden_residual(stack,parts,residual,correction_count,bias,
                    chunk_k=chunk_k,debug_poison=debug_fill is not None)
    with torch.cuda.device(value.device):''')
    producer=one(producer,'    outputs=tuple(empty_geglu((rows,channels),device=high.device,dtype=torch.float16) for _ in range(2))',
        '''    import os
    paired_mode=os.environ.get('UNIFOLM_DECISION_PAIRED_VENDOR','0')
    if paired_mode not in {'0','1'}:
        raise ValueError('Invalid decision paired vendor mode')
    paired_selected=(paired_mode=='1' and rows==10240
        and os.environ.get('UNIFOLM_DECISION_VENDOR_CORRECTION','0')=='1'
        and torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
    if paired_selected:
        stack=empty_geglu((2,rows,channels),device=high.device,dtype=torch.float16)
        outputs=tuple(stack.unbind(0));outputs[0]._paired_vendor_stack=stack
    else:
        outputs=tuple(empty_geglu((rows,channels),device=high.device,dtype=torch.float16) for _ in range(2))''')
    text=prefix+consumer+producer;ast.parse(text);return text


def paired_integration():
    return '''    decision_paired_vendor = os.environ.get("UNIFOLM_DECISION_PAIRED_VENDOR", "0")
    if decision_paired_vendor not in {"0", "1"}:
        raise ValueError("Invalid decision paired vendor mode")
    if decision_paired_vendor == "1":
        if os.environ.get("UNIFOLM_DECISION_VENDOR_CORRECTION", "0") != "1":
            raise ValueError("Paired vendor requires the accepted correction predicate")
        if sum(bool(getattr(m, "_decision_chunked_hidden_parts_installed", False))
                for m in model.model.diffusion_model.modules()) != 10:
            raise ValueError("Paired vendor requires the ten decision C640 blocks")
        print("DECISION_PAIRED_VENDOR_MODEL mode=1 blocks=10 M=10240 N=640 K=2560 chunk=512 "
              "GEGLU_output=fresh_two_plane_allocation activation_copy=none "
              "first_two_products=single_row_paired_cuBLAS third_product=device_predicated "
              "FP32_reduce_bias_residual=previous missing_layout_predicate_and_other_backend=previous "
              "weights=previous_first_timed_refresh world=previous", flush=True)
'''


def transform(original):
    text=paired_transform(original)
    assert text.count('from .paired_vendor_hidden_pipeline import paired_hidden_residual')==1
    text=text.replace('from .paired_vendor_hidden_pipeline import paired_hidden_residual',
        'from .paired_fused_vendor_hidden_pipeline import paired_hidden_residual')
    text=text.replace('UNIFOLM_DECISION_PAIRED_VENDOR','UNIFOLM_DECISION_PAIRED_FUSED_VENDOR')
    ast.parse(text);return text


def integration():
    return (paired_integration().replace('UNIFOLM_DECISION_PAIRED_VENDOR','UNIFOLM_DECISION_PAIRED_FUSED_VENDOR')
        .replace('DECISION_PAIRED_VENDOR_MODEL','DECISION_PAIRED_FUSED_VENDOR_MODEL')
        .replace('third_product=device_predicated ', 'third_product_and_FP32_reduce_bias_residual=fused_device_predicate '))
