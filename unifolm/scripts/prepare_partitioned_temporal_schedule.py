"""Tune only the static two-product launch, keeping the general path unchanged."""
import ast


def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def transform(original):
    text=one(original,'    batch,channels,frames,height,width=value.shape', '''    import os
    schedule=os.environ.get('UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE','2')
    if schedule not in {'2','3'}:
        raise ValueError('Invalid partitioned temporal schedule')
    batch,channels,frames,height,width=value.shape''')
    text=one(text,'    kernels=[]', '''    eligible=(batch==1 and frames==16 and outputs==channels
        and (channels,height,width) in {(320,40,64),(640,20,32)}
        and block_m==64 and block_n==64 and num_stages==2
        and group_m==(8 if channels==320 else 16)
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cuda.matmul.allow_tf32)
    kernels=[]''')
    text=one(text,'    for three in [False,True]:', '''    for three in [False,True]:
        tuned=schedule=='3' and eligible and not three
        selected_group=32 if tuned and channels==320 else group_m
        selected_stages=3 if tuned else num_stages''')
    text=one(text,'block_m,block_n,64,group_m,three,num_warps=4,num_stages=num_stages,',
             'block_m,block_n,64,selected_group,three,num_warps=4,num_stages=selected_stages,')
    old_tree,new_tree=ast.parse(original),ast.parse(text)
    old_jits=[n for n in old_tree.body if isinstance(n,ast.FunctionDef) and any(ast.unparse(d)=='triton.jit' for d in n.decorator_list)]
    new_jits=[n for n in new_tree.body if isinstance(n,ast.FunctionDef) and any(ast.unparse(d)=='triton.jit' for d in n.decorator_list)]
    assert len(old_jits)==len(new_jits)==1
    assert ast.dump(old_jits[0],include_attributes=True)==ast.dump(new_jits[0],include_attributes=True)
    return text


def integration():
    return '''    partitioned_temporal_schedule = os.environ.get("UNIFOLM_WORLD_PARTITIONED_TEMPORAL_SCHEDULE", "2")
    if partitioned_temporal_schedule not in {"2", "3"}:
        raise ValueError("Invalid partitioned temporal schedule")
    if partitioned_temporal_schedule == "3":
        if os.environ.get("UNIFOLM_WORLD_PARTITIONED_TEMPORAL", "0") != "1":
            raise ValueError("Temporal scheduling requires partitioned temporal kernels")
        partitioned_schedule_modules = [m for m in model.model.diffusion_model.modules()
            if getattr(m, "_world_partitioned_temporal_installed", False)]
        if (len(partitioned_schedule_modules) != 40
                or sum(m.in_channels == 320 for m in partitioned_schedule_modules) != 20
                or sum(m.in_channels == 640 for m in partitioned_schedule_modules) != 20):
            raise RuntimeError("Expected 20 C320 and 20 C640 partitioned temporal modules")
        print("WORLD_PARTITIONED_TEMPORAL_SCHEDULE_MODEL mode=3 blocks=40 C320_group=32 "
              "C640_group=16 two_product_stages=3 three_product_stages=2 "
              "scope=actual_world_narrow_shapes general_path_and_backend_fallback=previous "
              "JIT_math_and_cache=previous", flush=True)
'''
