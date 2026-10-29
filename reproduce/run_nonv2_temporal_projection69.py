"""Rebuild and qualify the projection boundary against original63 on two GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('decision', 'world'), required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--storage-config', type=Path)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    devices = a.gpus.split(',')
    assert len(devices) == len(set(devices)) == 2 and set(devices) <= set('0123')
    repo = Path(__file__).resolve().parents[1]
    base = repo/'experiments/nonv2-vae-all-display/unifolm'
    frozen = repo/'experiments/nonv2-temporal-projection69'/(a.stage+'-model-r1')
    identity = json.loads((frozen/'baseline-identity.json').read_text())
    hashes = json.loads((frozen/'source-hashes.json').read_text())
    work = a.workdir.resolve()
    assert not work.exists()
    work.mkdir(parents=True, mode=0o700)
    source = work/'source'
    source.mkdir()
    for name in ('src', 'configs', 'scripts/evaluation'):
        shutil.copytree(base/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('scripts/run_compute_benchmark.sh', 'scripts/team07_env.sh'):
        shutil.copyfile(base/name, source/name)
    manifest = identity['manifest']
    assert len(manifest) == 59
    manifest_file = base/identity['manifest_name']
    assert hashlib.sha256(manifest_file.read_bytes()).hexdigest() == identity['manifest_SHA256']
    shutil.copyfile(manifest_file, source/identity['manifest_name'])
    for name, digest in dict(manifest, **identity['assets']).items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    diagnostic = work/'diagnostic'
    diagnostic.mkdir()
    (diagnostic/'tmp').mkdir()
    for name, digest in hashes.items():
        data = (frozen/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        target = diagnostic/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (diagnostic/'scripts/evaluation').symlink_to(source/'scripts/evaluation', target_is_directory=True)
    if a.prepare_only:
        print(json.dumps(dict(prepared=True, no_GPU_execution=True, base_source_count=59,
            installed_baseline='qualified_original63', stage=a.stage, source=str(source),
            diagnostic=str(diagnostic))))
        return
    assert a.storage_config and a.cache_dir
    config = json.loads(a.storage_config.read_text())
    qualified = json.loads((repo/'experiments/private-nccl-tmpfs/qualified-adapter.json').read_text())
    assert config['cpu_cross_process_passed'] and config['source_sha256'] == qualified['source_sha256']
    library = Path(config['preload']).resolve()
    shared = Path(config['shared_root']).resolve()
    assert hashlib.sha256(library.read_bytes()).hexdigest() == config['preload_sha256']
    assert shared.is_dir() and shared.stat().st_uid == os.getuid() and shared.stat().st_mode & 0o777 == 0o700
    assert source.stat().st_dev == shared.stat().st_dev
    cache = a.cache_dir.resolve()
    assert cache.is_dir()
    env = {k:v for k,v in os.environ.items() if not k.startswith(('UNIFOLM_', 'NCCL_'))
           and k not in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'TRITON_CACHE_DIR', 'TRITON_CACHE_MANAGER',
                         'LD_PRELOAD', 'TEAM07_NCCL_SHM_ROOT')}
    env.update(CUDA_VISIBLE_DEVICES=a.gpus, MONITORED_GPUS=a.gpus, CANONICAL_ROOT=str(source),
        NONV2_FFN_CACHE_DIR=str(cache), LD_PRELOAD=str(library), TEAM07_NCCL_SHM_ROOT=str(shared),
        PYTHONPATH=os.pathsep.join(map(str, (diagnostic/'scripts', source/'src', source/'scripts', diagnostic))),
        PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='123', TMPDIR=str(diagnostic/'tmp'),
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
        HF_HUB_CACHE='/share/huggingface/hub', HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for name in ('check_nonv2_ffn59_cpu.py', 'check_nonv2_temporal_block59_cpu.py',
                 'check_nonv2_all_blocks59_cpu.py', 'check_nonv2_temporal_projection69_cpu.py'):
        subprocess.run([sys.executable, str(diagnostic/'scripts'/name)], cwd=diagnostic,
                       env=dict(env, CUDA_VISIBLE_DEVICES=''), check=True)
    subprocess.run([sys.executable, str(diagnostic/'retry_nccl_model_pair.py')],
                   cwd=diagnostic, env=env, check=True)
    print((diagnostic/'nccl-geglu-model-report.json').read_text())


if __name__ == '__main__':
    main()
