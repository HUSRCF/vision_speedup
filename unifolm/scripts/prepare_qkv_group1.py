"""Only change CTA traversal in the accepted zero-low QKV Python launcher."""
import ast


def transform(original):
    anchor='    _zero_low_gemm[('
    assert original.count(anchor)==1
    text=original.replace(anchor,'''    qkv_group=os.environ.get('UNIFOLM_WORLD_ZERO_QKV_GROUP','32')
    if qkv_group not in {'32','1-large'}:
        raise ValueError('Invalid zero-low QKV group')
    selected_group=(1 if qkv_group=='1-large' and output_groups==3 and bias is None
        and (rows,channels,inner) in large_qkv
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cuda.matmul.allow_tf32 else 32)
'''+anchor)
    anchor='rows, channels, inner, bias is not None, bm, 32, output_groups,'
    assert text.count(anchor)==1
    text=text.replace(anchor,'rows, channels, inner, bias is not None, bm, selected_group, output_groups,')
    ast.parse(text)
    # The host is below both JIT definitions, whose bodies and line numbers stay fixed.
    for a,b in zip(ast.parse(original).body,ast.parse(text).body):
        if isinstance(a,ast.FunctionDef) and any(ast.unparse(d)=='triton.jit' for d in a.decorator_list):
            assert ast.dump(a,include_attributes=True)==ast.dump(b,include_attributes=True)
    return text


def integration():
    return '''    zero_qkv_group = os.environ.get("UNIFOLM_WORLD_ZERO_QKV_GROUP", "32")
    if zero_qkv_group not in {"32", "1-large"}:
        raise ValueError("Invalid zero-low QKV group")
    if zero_qkv_group == "1-large":
        if os.environ.get("UNIFOLM_WORLD_ZERO_WEIGHT_LOW_QKV", "0") != "all":
            raise ValueError("QKV traversal tuning requires all world zero-low QKV wrappers")
        grouped_qkv_count = sum(getattr(m, "_world_zero_weight_low_qkv_installed", False)
            for m in model.model.diffusion_model.modules())
        if grouped_qkv_count != 48:
            raise RuntimeError("Expected 48 world zero-low QKV wrappers")
        print("WORLD_ZERO_QKV_GROUP_MODEL mode=1-large installed=48 eligible_world_calls=35 "
              "retained_small_world_calls=3 group_large=1 group_small=32 "
              "scope=world_three_actual_large_shapes FP32_math_cache_and_stages=previous "
              "decision=previous", flush=True)
'''
