"""Extend the qualified timed lifecycle without changing its accepted variant."""
import ast
from pathlib import Path
from nccl_parallel209_sources import production_sources as original_sources

def production_sources(base,variant):
    assert variant=='world-spatial-temporal'
    files,proof=original_sources(base,'world-ffn-temporal')
    name='src/unifolm_wma/utils/nccl_parallel_runtime209.py'
    runtime=files[name]
    changes=[
        ("assert value in ('0','world-ffn','both-ffn','world-ffn-temporal')",
         "assert value in ('0','world-ffn','both-ffn','world-ffn-temporal','world-spatial-temporal')"),
        ("    torch.cuda.synchronize();dist.barrier()\n    print('NCCL_PARALLEL_INITIALIZED '",
         "    if requested()=='world-spatial-temporal':\n        from . import nccl_spatial_temporal209_mode\n    torch.cuda.synchronize();dist.barrier()\n    print('NCCL_PARALLEL_INITIALIZED '"),
        ("    if choice=='world-ffn-temporal':\n        from .nccl_temporal_209_mode import install,mode as parallel_mode",
         "    if choice=='world-spatial-temporal':\n        from .nccl_spatial_temporal209_mode import install,mode as parallel_mode\n    elif choice=='world-ffn-temporal':\n        from .nccl_temporal_209_mode import install,mode as parallel_mode"),
        ("    if requested()=='world-ffn-temporal':assert report['temporal_captured']==88",
         "    if requested() in ('world-ffn-temporal','world-spatial-temporal'):assert report['temporal_captured']==88\n    if requested()=='world-spatial-temporal':assert (report['spatial_captured'],report['ffn_captured'])==(16,16)"),
    ]
    for old,new in changes:
        assert runtime.count(old)==1,old
        runtime=runtime.replace(old,new)
    files[name]=runtime
    for leaf in ('nccl_spatial_logical209_mode.py','nccl_spatial_temporal209_mode.py'):
        text=Path(__file__).with_name(leaf).read_text()
        text=text.replace('from nccl_geglu_209_mode import','from .nccl_geglu_209_mode import')
        for module in ('nccl_spatial_logical209_mode','nccl_temporal_209_mode','nccl_ffn_209_mode'):
            text=text.replace('import '+module+' as ','from . import '+module+' as ')
        files['src/unifolm_wma/utils/'+leaf]=text
    for text in files.values():ast.parse(text)
    return files,proof
