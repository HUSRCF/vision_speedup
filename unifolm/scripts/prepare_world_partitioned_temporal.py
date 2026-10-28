"""Preserve the accepted temporal cache and native fallback rules."""
import ast


def transform(original):
    text=original
    for old,new in [
        ('from .triton_zero_weight_low_temporal import zero_weight_low_temporal',
         'from .partitioned_temporal import partitioned_temporal_conv'),
        ("def make_forward(module, scope='all'):", "def make_forward(module, scope='implicit'):"),
        ("def install_world_zero_weight_low_temporal(diffusion, scope='all'):",
         "def install_world_partitioned_temporal(diffusion, scope='implicit'):"),
        ("    if scope not in {'all', 'implicit', 'packed'}:","    if scope != 'implicit':"),
        ("_world_zero_weight_low_temporal_installed","_world_partitioned_temporal_installed"),
        ('        return zero_weight_low_temporal(value, parts, count, module.bias)',
         "        return partitioned_temporal_conv(value, parts, count, module.bias,\n            group_m=8 if value.shape[1] == 320 else 16)")]:
        assert text.count(old)==1 if old.startswith('def ') or old.startswith('from ') or old.startswith('        return ') or old.startswith('    if scope ') else text.count(old)==2
        text=text.replace(old,new)
    ast.parse(text)
    return text


def integration():
    return '''    partitioned_temporal_mode = os.environ.get("UNIFOLM_WORLD_PARTITIONED_TEMPORAL", "0")
    if partitioned_temporal_mode not in {"0", "1"}:
        raise ValueError("UNIFOLM_WORLD_PARTITIONED_TEMPORAL must be 0 or 1")
    if partitioned_temporal_mode == "1":
        from .world_partitioned_temporal import install_world_partitioned_temporal
        partitioned_temporal_count = install_world_partitioned_temporal(model.model.diffusion_model)
        if partitioned_temporal_count != 40:
            raise RuntimeError("Expected 40 narrow temporal convolution modules")
        print("WORLD_PARTITIONED_TEMPORAL_MODEL mode=1 blocks=40 C320=20 C640=20 "
              "scope=world_narrow_implicit device_predicated_launches=2 logical_BK=64 "
              "FP32_sum_rounding_bias=previous weights=first_timed_warm_version_refresh_same_storage "
              "C1280_and_decision=previous", flush=True)
'''
