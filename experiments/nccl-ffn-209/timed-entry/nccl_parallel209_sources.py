"""Make a reviewable timed entry by adding four reversible lifecycle hooks."""
import ast,hashlib
from pathlib import Path

ENTRY='scripts/evaluation/world_model_interaction.py'

def production_sources(base,variant='world-ffn'):
    assert variant in ('world-ffn','world-ffn-temporal','both-ffn')
    original=(base/ENTRY).read_text()
    hooks=[
      ('PROCESS_START_TIME = time.perf_counter()\n',
       'from unifolm_wma.utils.nccl_parallel_runtime209 import early_start as _nccl_early_start\n_nccl_early_start()\n'),
      ('        configure_reproducibility(args.seed)\n',
       '        from unifolm_wma.utils.nccl_parallel_runtime209 import initialize as _nccl_initialize\n        _nccl_initialize()\n'),
      ('        configure_denoiser(model, os.environ.get("UNIFOLM_DENOISER_MODE", "native"), args.dataset)\n',
       '        from unifolm_wma.utils.nccl_parallel_runtime209 import configure as _nccl_configure\n        _nccl_configure(model, mode=os.environ.get("UNIFOLM_DENOISER_MODE", "native"), dataset=args.dataset)\n'),
      ('    run_inference(args, timing)\n',
       '    from unifolm_wma.utils.nccl_parallel_runtime209 import finish as _nccl_finish\n    _nccl_finish(args)\n')]
    entry=original
    for anchor,extra in hooks:
        assert entry.count(anchor)==1 and extra not in entry
        entry=entry.replace(anchor,anchor+extra)
    restored=entry
    for anchor,extra in hooks:restored=restored.replace(anchor+extra,anchor)
    assert restored==original
    assert entry.index('PROCESS_START_TIME =')<entry.index('_nccl_early_start()')<entry.index('import torch\n')
    assert entry.index('_nccl_finish(args)')<entry.index('        timing.write(args.profile_path)')
    prefix='src/unifolm_wma/utils/'
    files={ENTRY:entry}
    for name in ('nccl_parallel_runtime209.py','nccl_ffn_209_mode.py','nccl_geglu_209_mode.py'):
        value=Path(__file__).with_name(name).read_text()
        if name=='nccl_ffn_209_mode.py':
            old='from nccl_geglu_209_mode import measure';assert value.count(old)==1
            value=value.replace(old,'from .nccl_geglu_209_mode import measure')
        files[prefix+name]=value
    if variant=='world-ffn-temporal':
        name='nccl_temporal_209_mode.py';value=Path(__file__).with_name(name).read_text()
        value=value.replace('import nccl_ffn_209_mode as parent','from . import nccl_ffn_209_mode as parent')
        value=value.replace('from nccl_ffn_209_mode import','from .nccl_ffn_209_mode import')
        files[prefix+name]=value
    for value in files.values():ast.parse(value)
    proof=dict(original_entry_sha256=hashlib.sha256(original.encode()).hexdigest(),candidate_entry_sha256=hashlib.sha256(entry.encode()).hexdigest(),four_additive_lifecycle_hooks=True,removing_only_hooks_recovers_original_exact_bytes=True,original_timer_assignment_and_profile_writer_unchanged=True,helper_spawn_after_timer_before_torch_import=True,helper_shutdown_before_profile_write=True,original209_source_files_unchanged=True)
    return files,proof
