"""Read only frozen, hash-checked production files; no GPU initialization."""
import hashlib,json
from pathlib import Path
def production_sources(base,variant='both-spatial-temporal'):
    assert variant=='both-spatial-temporal'
    here=Path(__file__).resolve().parent;base=Path(base)
    original=json.loads((base/'world-head-filter-cache-production-sources.json').read_text())
    assert len(original)==209 and all(hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in original.items())
    proof=json.loads((here/'timing-proof.json').read_text())
    path='scripts/evaluation/world_model_interaction.py'
    original_entry=(base/path).read_bytes()
    assert hashlib.sha256(original_entry).hexdigest()==proof['original_entry_sha256']
    entry=original_entry.decode()
    hooks=(
        ('PROCESS_START_TIME = time.perf_counter()\n',
         'from unifolm_wma.utils.nccl_parallel_runtime209 import early_start as _nccl_early_start\n_nccl_early_start()\n'),
        ('        configure_reproducibility(args.seed)\n',
         '        from unifolm_wma.utils.nccl_parallel_runtime209 import initialize as _nccl_initialize\n        _nccl_initialize()\n'),
        ('        configure_denoiser(model, os.environ.get("UNIFOLM_DENOISER_MODE", "native"), args.dataset)\n',
         '        from unifolm_wma.utils.nccl_parallel_runtime209 import configure as _nccl_configure\n        _nccl_configure(model, mode=os.environ.get("UNIFOLM_DENOISER_MODE", "native"), dataset=args.dataset)\n'),
        ('    run_inference(args, timing)\n',
         '    from unifolm_wma.utils.nccl_parallel_runtime209 import finish as _nccl_finish\n    _nccl_finish(args)\n'),
    )
    for anchor,hook in hooks:
        assert entry.count(anchor)==1 and hook not in entry
        entry=entry.replace(anchor,anchor+hook)
    files={path:entry}
    for name,digest in json.loads((here/'overlay-source-hashes.json').read_text()).items():
        raw=entry.encode() if name==path else (here/'production-overlay'/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest
        files[name]=raw.decode()
    assert hashlib.sha256(entry.encode()).hexdigest()==proof['candidate_entry_sha256']
    restored=entry
    for anchor,hook in hooks:
        assert restored.count(hook)==1
        restored=restored.replace(anchor+hook,anchor)
    assert restored.encode()==original_entry
    assert entry.index('PROCESS_START_TIME =')<entry.index('_nccl_early_start()')<entry.index('import torch\n')
    assert entry.index('_nccl_finish(args)')<entry.index('        timing.write(args.profile_path)')
    return files,proof
