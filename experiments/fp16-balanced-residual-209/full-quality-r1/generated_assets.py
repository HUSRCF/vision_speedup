"""Generate exact evaluator/launcher aliases from the unchanged public base."""
import hashlib,json
from pathlib import Path


def generated_assets(base):
    base=Path(base);here=Path(__file__).resolve().parent
    proof=json.loads((here/'generated-assets.json').read_text())
    for name,digest in proof['originals'].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest
    entry=(base/'scripts/evaluation/world_model_interaction.py').read_text()
    anchor='        configure_denoiser(model, os.environ.get("UNIFOLM_DENOISER_MODE", "native"), args.dataset)\n'
    hook='        import fp16_balanced_residual209_mode as _balanced_residual\n        _balanced_residual.install()\n        _balanced_residual.enabled=True\n'
    assert entry.count(anchor)==1 and hook not in entry
    entry=entry.replace(anchor,anchor+hook)
    assert entry.index('PROCESS_START_TIME =')<entry.index('import torch\n')<entry.index('_balanced_residual.install()')
    shell=(base/'scripts/run_compute_benchmark.sh').read_text().replace(
        'scripts/run_compute_benchmark.sh','scripts/run_balanced_compute_benchmark.sh').replace(
        'scripts/evaluation/world_model_interaction.py','scripts/evaluation/world_model_interaction_balanced.py')
    files={'scripts/evaluation/world_model_interaction_balanced.py':entry,
        'scripts/run_balanced_compute_benchmark.sh':shell}
    assert {n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()}==proof['generated']
    return files
