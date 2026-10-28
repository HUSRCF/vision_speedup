"""Generate exact measured aliases from the unchanged public base."""
import hashlib,json
from pathlib import Path


def generated_assets(base):
    base=Path(base);proof=json.loads(Path(__file__).with_name('generated-assets.json').read_text())
    for name,digest in proof['originals'].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest
    entry=(base/'scripts/evaluation/world_model_interaction.py').read_text()
    anchor='        configure_denoiser(model, os.environ.get("UNIFOLM_DENOISER_MODE", "native"), args.dataset)\n'
    hook='        import kahan_packed_current209_mode as _kahan_packed\n        _kahan_packed.install()\n        _kahan_packed.enabled=True\n'
    assert entry.count(anchor)==1 and hook not in entry
    entry=entry.replace(anchor,anchor+hook)
    assert entry.index('PROCESS_START_TIME =')<entry.index('import torch\n')<entry.index('_kahan_packed.install()')
    shell=(base/'scripts/run_compute_benchmark.sh').read_text().replace(
        'scripts/run_compute_benchmark.sh','scripts/run_kahan_compute_benchmark.sh').replace(
        'scripts/evaluation/world_model_interaction.py','scripts/evaluation/world_model_interaction_kahan.py')
    files={'scripts/evaluation/world_model_interaction_kahan.py':entry,
        'scripts/run_kahan_compute_benchmark.sh':shell}
    assert {n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()}==proof['generated']
    return files
