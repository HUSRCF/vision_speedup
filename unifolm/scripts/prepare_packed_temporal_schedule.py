"""Host-only schedules for the two original C1280 packed temporal shapes."""
import ast


def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def transform(original):
    cut=original.index('def zero_weight_low_temporal(')
    body=original[cut:]
    body=one(body,'    rows, spatial = batch * frames * height * width, height * width',
        '''    import os
    mode=os.environ.get('UNIFOLM_WORLD_PACKED_TEMPORAL_SCHEDULE','0')
    if mode not in {'0','1'}:
        raise ValueError('Invalid packed temporal schedule')
    selected=(mode=='1' and implementation=='packed' and (batch,channels,frames,outputs)==(1,1280,16,1280)
        and (height,width) in {(10,16),(5,8)}
        and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
    rows, spatial = batch * frames * height * width, height * width''')
    body=one(body,"    group = (8 if channels == 320 else 16) if implementation == 'implicit' else 32",
        "    group = (8 if channels == 320 else 16) if implementation == 'implicit' else 32\n    if selected:group=8 if spatial==160 else 32\n    stages=3 if selected else 2")
    body=one(body,"implementation == 'implicit', group, num_warps=4, num_stages=2,",
        "implementation == 'implicit', group, num_warps=4, num_stages=stages,")
    text=original[:cut]+body;ast.parse(text);return text


def integration():
    return '''    packed_temporal_schedule = os.environ.get("UNIFOLM_WORLD_PACKED_TEMPORAL_SCHEDULE", "0")
    if packed_temporal_schedule not in {"0", "1"}:
        raise ValueError("Invalid packed temporal schedule")
    if packed_temporal_schedule == "1":
        count=sum(bool(getattr(m, "_world_zero_weight_low_temporal_installed", False))
            and getattr(m, "in_channels", None)==1280 for m in model.model.diffusion_model.modules())
        if count != 48:
            raise ValueError("Packed temporal schedule requires the 48 original C1280 wrappers")
        print("WORLD_PACKED_TEMPORAL_SCHEDULE_MODEL mode=1 blocks=48 C=1280 M2560_group=8 M640_group=32 "
              "stages=3 BM=64 BN=64 warps=4 logical_BK=64 "
              "JIT_math_device_predicate_packing_output_and_cache=previous narrow_and_decision=previous", flush=True)
'''
