"""Explicit world-only K32 dispatch on the current accepted producer hosts."""


def transform(text,width):
    assert width in {320,640}
    assert '_geglu_subtile32' not in text
    if 'import os\n' not in text:
        anchor='import torch\n';assert text.count(anchor)==1
        text=text.replace(anchor,'import os\n'+anchor)
    anchor='from .triton_geglu_hidden_parts_vector_store import _zero_geglu_hidden_parts_vector_store as _zero_geglu_hidden_parts\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'from .triton_geglu_subtile32 import _geglu_subtile32\n')
    split=text.index('def packed_hidden_ffn320_residual(' if width==320 else 'def packed_hidden_ffn_residual(')
    prefix,suffix=text[:split],text[split:]
    anchor='    outputs=tuple(empty_geglu((rows,channels),device=high.device,dtype=torch.float16) for _ in range(2))'
    assert prefix.count(anchor)==1
    geometry=(40960,1280,320) if width==320 else (10240,2560,640)
    prefix=prefix.replace(anchor,f'''    subtile=os.environ.get('UNIFOLM_WORLD_GEGLU_SUBTILE32','0')
    if subtile not in {{'0','1'}}:raise ValueError('Invalid world GEGLU K32 mode')
    use_subtile=(subtile=='1' and (rows,channels,inner)=={geometry!r}
                 and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
    kernel=_geglu_subtile32 if use_subtile else _zero_geglu_hidden_parts
'''+anchor)
    anchor='    _zero_geglu_hidden_parts[('
    assert prefix.count(anchor)==1
    return prefix.replace(anchor,'    kernel[(')+suffix


def integration():
    return '''    geglu_subtile32=os.environ.get('UNIFOLM_WORLD_GEGLU_SUBTILE32','0')
    if geglu_subtile32 not in {'0','1'}:raise ValueError('Invalid world GEGLU K32 mode')
    subtile320_count=sum(bool(getattr(m,'_world_ffn320_zero_installed',False))
        for m in model.model.diffusion_model.modules())
    subtile640_count=sum(bool(getattr(m,'_world_geglu_hidden_parts_installed',False))
        for m in model.model.diffusion_model.modules())
    if geglu_subtile32=='1' and (subtile320_count!=10 or subtile640_count!=10 or
            os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS')!='1' or
            os.environ.get('UNIFOLM_WORLD_GEGLU_HIDDEN_PARTS_C320')!='1' or
            os.environ.get('UNIFOLM_WORLD_FFN320_ZERO_RESIDUAL')!='1' or
            os.environ.get('UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU')!='all'):
        raise ValueError('World GEGLU K32 requires the accepted twenty C320/C640 blocks')
    print(f'WORLD_GEGLU_SUBTILE32_MODEL mode={geglu_subtile32} blocks={subtile320_count+subtile640_count} '
          f'C320={subtile320_count} C640={subtile640_count} '
          'logical_BK=64 physical_BK=32 scope=two_actual_world_shapes '
          'FP32_sum_rounding_bias_GELU=previous hidden=fresh_Half_high_low '
          'device_predicate=previous_two_or_three_products '
          'weights=previous_first_timed_warm_version_refresh_same_storage '
          'other_shapes_backend_and_decision=previous C1280=previous',flush=True)
'''
