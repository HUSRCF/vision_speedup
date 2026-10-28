"""Only host tile selection for the accepted C1280 K32 implementation."""
import ast


def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def transform(original):
    cut=original.index('def packed_hidden_ffn1280_residual(')
    prefix,suffix=original[:cut],original[cut:]
    prefix=one(prefix,'    kernel=_both_geglu_subtile32 if subtile_selected else _zero_geglu_hidden_parts',
        '''    tiles_mode=os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32_TILES','0')
    if tiles_mode not in {'0','1'}:
        raise ValueError('Invalid C1280 GEGLU subtile tile mode')
    tuned_tiles=tiles_mode=='1' and subtile_selected
    bm=(64 if rows==2560 else 32) if tuned_tiles else 64
    bn=64 if tuned_tiles else 32
    group=16 if tuned_tiles else 32
    kernel=_both_geglu_subtile32 if subtile_selected else _zero_geglu_hidden_parts''')
    prefix=one(prefix,'    kernel[(triton.cdiv(rows,64)*triton.cdiv(channels,32),)](',
        '    kernel[(triton.cdiv(rows,bm)*triton.cdiv(channels,bn),)](')
    prefix=one(prefix,'rows,channels,inner,bias is not None,64,32,64,32,False,False,False,',
        'rows,channels,inner,bias is not None,bm,bn,64,group,False,False,False,')
    text=prefix+suffix;ast.parse(text);return text


def integration():
    return '''    geglu_c1280_tiles = os.environ.get("UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32_TILES", "0")
    if geglu_c1280_tiles not in {"0", "1"}:
        raise ValueError("Invalid C1280 GEGLU subtile tile mode")
    if geglu_c1280_tiles == "1":
        if os.environ.get("UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32", "0") != "1":
            raise ValueError("C1280 tile schedule requires the accepted subtile mode")
        if sum(bool(getattr(m, "_world_geglu_hidden_parts_c1280_zero_installed", False))
                for m in model.model.diffusion_model.modules()) != 12:
            raise ValueError("C1280 tile schedule requires the twelve world blocks")
        print("WORLD_GEGLU_C1280_SUBTILE32_TILES_MODEL mode=1 blocks=12 "
              "M2560_BM=64 M640_BM=32 BN=64 group=16 stages=2 "
              "logical_BK=64 physical_BK=32 inherited_subtile_marker=default_geometry "
              "other_shapes_and_decision=previous JIT_math_and_cache=previous", flush=True)
'''
