"""Change only post-JIT host scheduling for the two world attention projections."""
import ast


def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def transform(original):
    cut=original.index('def packed_attention_output(')
    body=original[cut:]
    body=one(body,"    output=empty_gemm('classic',(batch,length,channels),device=high.device,dtype=torch.float32)",
        '''    import os
    mode=os.environ.get('UNIFOLM_WORLD_ATTENTION_PROJECTION_SCHEDULE','0')
    if mode not in {'0','1'}:
        raise ValueError('Invalid attention projection schedule')
    selected=(mode=='1' and not torch.is_grad_enabled()
        and (batch,length,heads,width,channels) in {(16,2560,5,64,320),(16,640,10,64,640)}
        and bias is not None and bias.dtype==torch.float32 and bias.device==high.device and bias.is_contiguous()
        and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
    group,stages=(8,3) if selected else (32,2)
    output=empty_gemm('classic',(batch,length,channels),device=high.device,dtype=torch.float32)''')
    body=one(body,'        64,64,64,32,num_warps=4,num_stages=2,enable_fp_fusion=False)',
        '        64,64,64,group,num_warps=4,num_stages=stages,enable_fp_fusion=False)')
    result=original[:cut]+body;ast.parse(result);return result


def integration():
    return '''    attention_projection_schedule = os.environ.get("UNIFOLM_WORLD_ATTENTION_PROJECTION_SCHEDULE", "0")
    if attention_projection_schedule not in {"0", "1"}:
        raise ValueError("Invalid attention projection schedule")
    if attention_projection_schedule == "1":
        count=sum(bool(getattr(m, "_world_attention_hidden_projection_installed", False))
            for m in model.model.diffusion_model.modules())
        if count != 10:
            raise ValueError("Attention projection schedule requires the ten original world wrappers")
        print("WORLD_ATTENTION_PROJECTION_SCHEDULE_MODEL mode=1 blocks=10 C320=5 C640=5 "
              "group=8 stages=3 BM=64 BN=64 BK=64 warps=4 "
              "JIT_math_cache_input_parts_and_FP32_output=previous decision_and_other_shapes=previous", flush=True)
'''
