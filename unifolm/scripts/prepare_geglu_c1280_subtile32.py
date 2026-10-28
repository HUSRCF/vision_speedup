"""Opt in only the two real C1280 world producer shapes; preserve cache owners."""
import ast


def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def transform(original):
    text=one(original,'from .triton_geglu_hidden_parts_vector_store import _zero_geglu_hidden_parts_vector_store as _zero_geglu_hidden_parts',
        'from .triton_geglu_hidden_parts_vector_store import _zero_geglu_hidden_parts_vector_store as _zero_geglu_hidden_parts\nfrom .triton_geglu_both_subtile32 import _both_geglu_subtile32')
    cut=text.index('def packed_hidden_ffn1280_residual(');prefix,suffix=text[:cut],text[cut:]
    prefix=one(prefix,'    outputs=tuple(empty_geglu', '''    subtile_mode=os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32','0')
    if subtile_mode not in {'0','1'}:
        raise ValueError('Invalid C1280 GEGLU subtile mode')
    subtile_selected=(subtile_mode=='1' and rows in {2560,640}
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cuda.matmul.allow_tf32)
    kernel=_both_geglu_subtile32 if subtile_selected else _zero_geglu_hidden_parts
    outputs=tuple(empty_geglu''')
    prefix=one(prefix,'    _zero_geglu_hidden_parts[(','    kernel[(')
    prefix=one(prefix,"num_warps=4,num_stages=int(os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_STAGES','2')),enable_fp_fusion=False)",
        "num_warps=4,num_stages=2 if subtile_selected else int(os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_STAGES','2')),enable_fp_fusion=False)")
    result=prefix+suffix;ast.parse(result);return result


def integration():
    return '''    geglu_c1280_subtile = os.environ.get("UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32", "0")
    if geglu_c1280_subtile not in {"0", "1"}:
        raise ValueError("Invalid C1280 GEGLU subtile mode")
    if geglu_c1280_subtile == "1":
        geglu_c1280_subtile_blocks = sum(bool(getattr(m, "_world_geglu_hidden_parts_c1280_zero_installed", False))
            for m in model.model.diffusion_model.modules())
        if geglu_c1280_subtile_blocks != 12:
            raise ValueError("C1280 subtile kernels require the twelve world zero-residual blocks")
        print("WORLD_GEGLU_C1280_SUBTILE32_MODEL mode=1 blocks=12 logical_BK=64 physical_BK=32 "
              "stages=2 BM=64 BN=32 group=32 scope=actual_M2560_640_world_only "
              "both_value_and_gate_short_sums=previous FP32_bias_GELU_and_cache=previous "
              "other_shapes_and_decision=previous", flush=True)
'''
